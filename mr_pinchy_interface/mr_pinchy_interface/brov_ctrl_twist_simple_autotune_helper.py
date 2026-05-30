#!/usr/bin/env python3
"""Autotune helper for ``brov_ctrl_twist_simple``.

Run this node *instead of* ``brov_ctrl_twist_simple`` while tuning. It drives
its own velocity-reference schedule (one axis at a time), runs the identical
feed-forward + PID control law, owns the ``/mavros/manual_control/send`` stream
(so the vehicle never disarms), and logs everything needed to pick kff/kp/ki
to a CSV file.

Recipe:
  1. Run with ``feedforward_only:=true`` and kp=ki=kd=0. For each held step,
     read the steady-state measured velocity from the CSV. The static gain is
     ``measured_vel / output_norm``; set ``kff = 1 / static_gain``.
  2. Re-run closed-loop, raise kp until responsive without oscillation, then add
     ki to remove residual steady-state error. Inspect the per-term columns and
     the ref-vs-meas tracking in the CSV.

Safety: reference magnitudes are capped (``max_lin_vel`` / ``max_yaw_vel``) to
keep travel inside the tank, and the node neutralises + pauses the schedule if
the vehicle leaves ``tank_range``.
"""
from __future__ import annotations

import copy
import csv
import os
from datetime import datetime

import numpy as np
import rclpy
from mavros_msgs.msg import ManualControl
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from tf2_geometry_msgs import do_transform_vector3
from tf2_ros import Buffer, TransformException, TransformListener

from mr_pinchy_interface.brov_utils import vector3_stamped, vector3_to_array

AXIS_NAMES = ("surge", "sway", "heave", "yaw")


def normalize_tf_frame_id(frame_id: str) -> str:
    return str(frame_id).lstrip("/")


class BrovTwistSimpleAutotuneHelper(Node):
    def __init__(self) -> None:
        super().__init__("brov_ctrl_twist_simple_autotune_helper")

        self.declare_parameter("odom_topic", "/mocap/mr_pinchy/odom_ekf")
        self.declare_parameter("manual_control_topic", "/mavros/manual_control/send")
        self.declare_parameter("base_link_frame", "mr_pinchy/base_link_ekf")
        self.declare_parameter("csv_path", "")
        self.declare_parameter("feedforward_only", False)
        # Reference safety caps (keep travel inside the tank).
        self.declare_parameter("max_lin_vel", 0.2)
        self.declare_parameter("max_yaw_vel", 0.4)
        # Schedule timing.
        self.declare_parameter("settle_sec", 4.0)
        self.declare_parameter("hold_sec", 6.0)
        # Fractions of the per-axis cap to command, in order, for every axis.
        self.declare_parameter("step_levels", [0.5, 1.0, -0.5, -1.0])
        # Axes to exercise (subset of surge/sway/heave/yaw).
        self.declare_parameter("axes", list(AXIS_NAMES))

        odom_topic = str(self.get_parameter("odom_topic").value)
        mc_topic = str(self.get_parameter("manual_control_topic").value)
        self._base_link_frame = normalize_tf_frame_id(
            self.get_parameter("base_link_frame").value
        )
        self._feedforward_only = bool(self.get_parameter("feedforward_only").value)
        self._max_lin_vel = float(self.get_parameter("max_lin_vel").value)
        self._max_yaw_vel = float(self.get_parameter("max_yaw_vel").value)
        self._settle_sec = float(self.get_parameter("settle_sec").value)
        self._hold_sec = float(self.get_parameter("hold_sec").value)
        step_levels = [float(v) for v in self.get_parameter("step_levels").value]
        axes = [str(a) for a in self.get_parameter("axes").value]

        self._publish_rate_hz = 50.0
        self._odom_timeout_sec = 0.3
        self._tf_timeout = Duration(seconds=0.05)
        self._tank_margin = 0.3  # neutralise this far from a wall

        # Control law (kept identical to brov_ctrl_twist_simple so the recorded
        # behaviour matches the real controller). Tweak these as you tune.
        self._kp = np.array([7.0, 8.0, 2.5, 1.4], dtype=float)
        self._ki = np.array([0.5, 0.5, 0.1, 0.3], dtype=float)
        self._kd = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self._integral_limit = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        self._kff = np.array([1.0 / 0.3, 1.0 / 0.25, 1.0 / 0.3, 1.0 / 0.65], dtype=float)
        if self._feedforward_only:
            self._kp[:] = 0.0
            self._ki[:] = 0.0
            self._kd[:] = 0.0
        self._integral = np.zeros(4, dtype=float)
        self._alpha = np.array([0.7, 0.7, 0.7, 0.7], dtype=float)

        # Tank safety bounds: [[x_min, y_min, z_min], [x_max, y_max, z_max]]
        self._tank_range = np.array([[1.0, -1.8, 0.5], [6.0, 1.8, 3.0]])

        caps = np.array(
            [self._max_lin_vel, self._max_lin_vel, self._max_lin_vel, self._max_yaw_vel],
            dtype=float,
        )
        self._schedule = self._build_schedule(axes, step_levels, caps)
        self._phase_idx = 0
        self._phase_started_sec: float | None = None

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._mc_pub = self.create_publisher(ManualControl, mc_topic, qos)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._last_odom: Odometry | None = None
        self._last_odom_rx_sec: float = 0.0
        self._prev_meas: np.ndarray | None = None
        self._prev_feedback_sec: float | None = None
        self._last_ctrl_cmd = np.array([0.0, 0.0, 500.0, 0.0], dtype=float)
        self._start_sec = self.get_clock().now().nanoseconds * 1e-9

        self._csv_path = self._resolve_csv_path(str(self.get_parameter("csv_path").value))
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(self._csv_header())

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)
        self.get_logger().info(
            f"autotune_helper: odom='{odom_topic}', mc='{mc_topic}', "
            f"feedforward_only={self._feedforward_only}, "
            f"caps(lin={self._max_lin_vel:.2f}, yaw={self._max_yaw_vel:.2f}), "
            f"phases={len(self._schedule)}, csv='{self._csv_path}'"
        )

    # ------------------------------------------------------------------ setup
    def _build_schedule(
        self, axes: list[str], step_levels: list[float], caps: np.ndarray
    ) -> list[dict]:
        """Return ordered phases. A settle (zero) precedes every held step."""
        phases: list[dict] = [
            {"label": "settle", "axis": -1, "ref": np.zeros(4), "dur": self._settle_sec}
        ]
        for axis_name in axes:
            if axis_name not in AXIS_NAMES:
                self.get_logger().warn(f"Ignoring unknown axis '{axis_name}'")
                continue
            idx = AXIS_NAMES.index(axis_name)
            for level in step_levels:
                ref = np.zeros(4, dtype=float)
                ref[idx] = level * caps[idx]
                phases.append(
                    {
                        "label": f"{axis_name}_{level:+.2f}",
                        "axis": idx,
                        "ref": ref,
                        "dur": self._hold_sec,
                    }
                )
                phases.append(
                    {
                        "label": "settle",
                        "axis": -1,
                        "ref": np.zeros(4),
                        "dur": self._settle_sec,
                    }
                )
        return phases

    def _resolve_csv_path(self, requested: str) -> str:
        if requested:
            return os.path.expanduser(requested)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "ff" if self._feedforward_only else "pid"
        return os.path.join(
            os.path.expanduser("~"), f"brov_autotune_{mode}_{stamp}.csv"
        )

    @staticmethod
    def _csv_header() -> list[str]:
        cols = ["t", "phase", "axis"]
        for a in AXIS_NAMES:
            cols.append(f"ref_{a}")
        for a in AXIS_NAMES:
            cols.append(f"meas_{a}")
        for a in AXIS_NAMES:
            cols.append(f"err_{a}")
        for term in ("ff", "p", "i", "d"):
            for a in AXIS_NAMES:
                cols.append(f"{term}_{a}")
        cols += ["mc_x", "mc_y", "mc_z", "mc_r"]
        cols += ["pos_x", "pos_y", "pos_z"]
        cols += ["in_tank", "paused"]
        return cols

    # --------------------------------------------------------------- callbacks
    def _odom_cb(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_rx_sec = self.get_clock().now().nanoseconds * 1e-9

    # ----------------------------------------------------------- tf / odom utils
    def _transform_vector_to_base_link(self, msg, *, source_frame, stamp=None):
        source_frame = normalize_tf_frame_id(source_frame)
        if source_frame == self._base_link_frame:
            return vector3_to_array(msg)
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_link_frame, source_frame, rclpy.time.Time(),
                timeout=self._tf_timeout,
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {source_frame} -> {self._base_link_frame} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None
        vector = vector3_stamped(msg, frame_id=source_frame, stamp=stamp)
        transformed = do_transform_vector3(vector, copy.deepcopy(transform))
        return vector3_to_array(transformed.vector)

    def _get_body_velocity(self, odom: Odometry):
        source_frame = normalize_tf_frame_id(getattr(odom, "child_frame_id", ""))
        if not source_frame:
            return None
        stamp = getattr(getattr(odom, "header", None), "stamp", None)
        linear = self._transform_vector_to_base_link(
            odom.twist.twist.linear, source_frame=source_frame, stamp=stamp
        )
        angular = self._transform_vector_to_base_link(
            odom.twist.twist.angular, source_frame=source_frame, stamp=stamp
        )
        if linear is None or angular is None:
            return None
        return np.concatenate([linear, angular]).astype(float)

    @staticmethod
    def _get_position(odom: Odometry) -> np.ndarray:
        p = odom.pose.pose.position
        return np.array([p.x, p.y, p.z], dtype=float)

    # ---------------------------------------------------------------- publishing
    def _publish_manual_control(self, x, y, z, r) -> None:
        msg = ManualControl()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.x = float(x)
        msg.y = float(y)
        msg.z = float(z)
        msg.r = float(r)
        msg.buttons = 0
        self._mc_pub.publish(msg)

    def _publish_neutral(self) -> None:
        # Always keep the stream alive so the FCU does not disarm on stale input.
        self._publish_manual_control(0.0, 0.0, 500.0, 0.0)
        self._last_ctrl_cmd = np.array([0.0, 0.0, 500.0, 0.0], dtype=float)

    def _reset_pid(self) -> None:
        self._prev_meas = None
        self._prev_feedback_sec = None
        self._integral = np.zeros(4, dtype=float)

    # ----------------------------------------------------------------- schedule
    def _current_ref(self, now: float):
        if self._phase_idx >= len(self._schedule):
            return None, None
        phase = self._schedule[self._phase_idx]
        if self._phase_started_sec is None:
            self._phase_started_sec = now
            self.get_logger().info(
                f"[{self._phase_idx + 1}/{len(self._schedule)}] phase '{phase['label']}'"
            )
        if (now - self._phase_started_sec) >= phase["dur"]:
            self._phase_idx += 1
            self._phase_started_sec = None
            # Reset integral between phases so each step starts clean.
            self._reset_pid()
            return self._current_ref(now)
        return phase["ref"], phase

    # --------------------------------------------------------------------- tick
    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9

        # Schedule finished: hold neutral forever (keeps vehicle armed).
        if self._phase_idx >= len(self._schedule):
            self._reset_pid()
            self._publish_neutral()
            self.get_logger().info(
                "Autotune schedule complete; holding neutral.",
                throttle_duration_sec=5.0,
            )
            return

        if self._last_odom is None or (now - self._last_odom_rx_sec) > self._odom_timeout_sec:
            self._reset_pid()
            self._publish_neutral()
            self.get_logger().warn(
                "Odometry timeout; holding neutral (schedule paused).",
                throttle_duration_sec=2.0,
            )
            return

        pos = self._get_position(self._last_odom)
        in_tank = bool(
            np.all(pos >= self._tank_range[0] + self._tank_margin)
            and np.all(pos <= self._tank_range[1] - self._tank_margin)
        )
        if not in_tank:
            # Pause the schedule (freeze phase timer) and neutralise until the
            # operator brings the vehicle back inside the safe envelope.
            self._phase_started_sec = None
            self._reset_pid()
            self._publish_neutral()
            self.get_logger().warn(
                f"Out of tank range pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}); "
                "neutral + paused.",
                throttle_duration_sec=1.0,
            )
            self._log_row(now, np.zeros(4), None, paused=True, in_tank=False, pos=pos)
            return

        nu = self._get_body_velocity(self._last_odom)
        if nu is None:
            self._reset_pid()
            self._publish_neutral()
            return

        ref, phase = self._current_ref(now)
        if ref is None:
            self._reset_pid()
            self._publish_neutral()
            return

        meas = np.array([nu[0], nu[1], nu[2], nu[5]], dtype=float)
        error = ref - meas

        if self._prev_meas is not None and self._prev_feedback_sec is not None:
            dt = now - self._prev_feedback_sec
            if dt > 1e-6:
                deriv = -(meas - self._prev_meas) / dt
                self._integral += error * dt
                self._integral = np.clip(
                    self._integral, -self._integral_limit, self._integral_limit
                )
            else:
                deriv = np.zeros(4, dtype=float)
        else:
            deriv = np.zeros(4, dtype=float)
        self._prev_meas = meas.copy()
        self._prev_feedback_sec = now

        ff_term = self._kff * ref
        p_term = self._kp * error
        i_term = self._ki * self._integral
        d_term = self._kd * deriv
        output = np.clip(ff_term + p_term + i_term + d_term, -1.0, 1.0)

        output_limit = 1000.0
        x = float(output[0]) * output_limit
        y = float(output[1]) * output_limit
        z = float(-output[2]) * output_limit / 2 + 500.0
        r = float(output[3]) * output_limit
        self._last_ctrl_cmd = (
            self._alpha * np.array([x, y, z, r])
            + (1 - self._alpha) * self._last_ctrl_cmd
        )
        self._publish_manual_control(*self._last_ctrl_cmd)

        self._log_row(
            now, ref, phase, paused=False, in_tank=True, pos=pos, meas=meas,
            error=error, ff=ff_term, p=p_term, i=i_term, d=d_term,
        )

        self.get_logger().info(
            f"[{phase['label']}] ref=({ref[0]:.2f},{ref[1]:.2f},{ref[2]:.2f},{ref[3]:.2f}) "
            f"meas=({meas[0]:.2f},{meas[1]:.2f},{meas[2]:.2f},{meas[3]:.2f}) "
            f"mc=(x={self._last_ctrl_cmd[0]:.0f},y={self._last_ctrl_cmd[1]:.0f},"
            f"z={self._last_ctrl_cmd[2]:.0f},r={self._last_ctrl_cmd[3]:.0f})",
            throttle_duration_sec=0.5,
        )

    # ------------------------------------------------------------------ logging
    def _log_row(
        self, now, ref, phase, *, paused, in_tank, pos,
        meas=None, error=None, ff=None, p=None, i=None, d=None,
    ) -> None:
        z4 = np.zeros(4, dtype=float)
        meas = z4 if meas is None else meas
        error = z4 if error is None else error
        ff = z4 if ff is None else ff
        p = z4 if p is None else p
        i = z4 if i is None else i
        d = z4 if d is None else d
        label = phase["label"] if phase is not None else ("paused" if paused else "idle")
        axis = phase["axis"] if phase is not None else -1
        row = [f"{now - self._start_sec:.4f}", label, axis]
        row += [f"{v:.4f}" for v in ref]
        row += [f"{v:.4f}" for v in meas]
        row += [f"{v:.4f}" for v in error]
        row += [f"{v:.4f}" for v in ff]
        row += [f"{v:.4f}" for v in p]
        row += [f"{v:.4f}" for v in i]
        row += [f"{v:.4f}" for v in d]
        row += [f"{v:.2f}" for v in self._last_ctrl_cmd]
        row += [f"{v:.4f}" for v in pos]
        row += [int(in_tank), int(paused)]
        self._csv.writerow(row)
        self._csv_file.flush()

    def destroy_node(self) -> bool:
        try:
            self._publish_neutral()
        except Exception:
            pass
        try:
            self._csv_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovTwistSimpleAutotuneHelper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
