#!/usr/bin/env python3
"""
ROS 2 node: read the same TTY keys as ``bluerov2_keyboard_teleop``, print them,
publish MAVROS RC override, and call the arming service.

- Publishes ``mavros_msgs/msg/OverrideRCIn`` on ``/mavros/rc/override`` (param ``rc_topic``).
- Calls ``mavros_msgs/srv/CommandBool`` on ``/mavros/cmd/arming``: **t** arm, **y** disarm.

Run in an interactive terminal (``ros2 run mr_pinchy_interface keyboard_echo``) and focus it.

Keys (same as teleop TTY mode):

  Motion: w/s, a/d, Space, z, q/e, g/v, x/c
  Other:  t / y (arm/disarm), p / n (widen/narrow PWM delta)
  Esc:    stop key reader and shut down the node
"""

from __future__ import annotations

import os
import queue
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from mavros_msgs.msg import OverrideRCIn
from mavros_msgs.srv import CommandBool

# Same mapping labels as bluerov2_keyboard_teleop (TTY backend)
_KEY_LABELS: dict[str, str] = {
    "w": "forward",
    "s": "backward",
    "a": "strafe_left",
    "d": "strafe_right",
    " ": "ascend",
    "z": "descend",
    "q": "yaw_left",
    "e": "yaw_right",
    "g": "pitch_up",
    "v": "pitch_down",
    "x": "roll_left",
    "c": "roll_right",
    "t": "arm (MAVROS service)",
    "y": "disarm (MAVROS service)",
    "p": "pwm_delta+ (RC override)",
    "n": "pwm_delta- (RC override)",
}

# Same motion tags as bluerov2_keyboard_teleop (TTY)
_TTY_KEY_TO_TAGS: dict[str, tuple[str, ...]] = {
    "w": ("fwd",),
    "s": ("back",),
    "a": ("left",),
    "d": ("right",),
    "q": ("yaw_left",),
    "e": ("yaw_right",),
    " ": ("up",),
    "z": ("down",),
    "g": ("pitch_up",),
    "v": ("pitch_down",),
    "x": ("roll_left",),
    "c": ("roll_right",),
}


def _consume_esc_sequence(fd: int) -> bool:
    """If CSI sequence, consume and return False. If bare ESC, return True (quit)."""
    r, _, _ = select.select([fd], [], [], 0.05)
    if not r:
        return True
    try:
        follow = os.read(fd, 2)
    except OSError:
        return True
    if follow.startswith(b"["):
        while True:
            r2, _, _ = select.select([fd], [], [], 0.02)
            if not r2:
                break
            try:
                tail = os.read(fd, 8)
            except OSError:
                break
            if not tail or tail[-1:] in (b"~", b"A", b"B", b"C", b"D"):
                break
        return False
    return True


def _tty_reader_loop(
    key_q: queue.Queue,
    stop: threading.Event,
    log_fn,
) -> None:
    fd = None
    tty_f = None
    old = None
    try:
        try:
            tty_f = open("/dev/tty", "rb", buffering=0)
            fd = tty_f.fileno()
        except OSError:
            if sys.stdin.isatty():
                fd = sys.stdin.fileno()
            else:
                log_fn(
                    "No /dev/tty and stdin is not a TTY — run in an interactive terminal."
                )
                return

        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        log_fn("Reading keys (focus this terminal). Esc alone = stop echo loop.")

        while not stop.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.2)
            except (ValueError, OSError):
                break
            if not r:
                continue
            try:
                chunk = os.read(fd, 64)
            except OSError:
                break
            if not chunk:
                break
            for b in chunk:
                if b == 0o33:
                    if _consume_esc_sequence(fd):
                        key_q.put(("esc",))
                    continue
                if b in (0o15, 0o12):
                    continue
                try:
                    ch = chr(b).lower()
                except ValueError:
                    continue
                if "a" <= ch <= "z" or ch == " ":
                    key_q.put(("char", ch))
    finally:
        if old is not None and fd is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except termios.error:
                pass
        if tty_f is not None:
            try:
                tty_f.close()
            except OSError:
                pass


class BrovKeyboardController(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_echo")

        self.declare_parameter("rc_topic", "/mavros/rc/override")
        self.declare_parameter("arming_service", "/mavros/cmd/arming")
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("neutral_pwm", 1500)
        self.declare_parameter("delta_pwm", 100)
        self.declare_parameter("key_idle_timeout_sec", 0.4)

        rc_topic = self.get_parameter("rc_topic").get_parameter_value().string_value
        arm_srv = self.get_parameter("arming_service").get_parameter_value().string_value
        rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        if rate_hz <= 0.0:
            rate_hz = 20.0

        self._neutral = int(
            self.get_parameter("neutral_pwm").get_parameter_value().integer_value
        )
        if self._neutral <= 0:
            self._neutral = 1500
        self._delta = int(
            self.get_parameter("delta_pwm").get_parameter_value().integer_value
        )
        if self._delta <= 0:
            self._delta = 100

        idle_timeout = float(
            self.get_parameter("key_idle_timeout_sec").get_parameter_value().double_value
        )
        if idle_timeout <= 0.0:
            idle_timeout = 0.4
        self._idle_timeout = idle_timeout

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._rc_pub = self.create_publisher(OverrideRCIn, rc_topic, qos)

        self._arming_client = self.create_client(CommandBool, arm_srv)
        self._armed = False
        self._arming_in_flight = False
        self._arming_service_name = arm_srv

        self._motion_lock = threading.Lock()
        self._last_refresh: dict[str, float] = {}

        self._key_q: queue.Queue = queue.Queue()
        self._stop_reader = threading.Event()
        self._prev_typed: set[str] = set()

        self._reader = threading.Thread(
            target=_tty_reader_loop,
            args=(self._key_q, self._stop_reader, self.get_logger().warn),
            daemon=True,
        )
        self._reader.start()

        period = 1.0 / rate_hz
        self.create_timer(period, self._tick)
        self.get_logger().info(
            f"keyboard_echo: publishing OverrideRCIn on '{rc_topic}' at {rate_hz:.1f} Hz; "
            f"arming '{arm_srv}' [t]/[y]; key_idle_timeout={idle_timeout:.2f}s"
        )

    def _call_arming(self, arm: bool) -> None:
        if self._arming_in_flight:
            print("[keyboard_echo] arming: request already in progress, ignored", flush=True)
            return
        if not self._arming_client.service_is_ready():
            self.get_logger().warn(
                f"Arming service '{self._arming_service_name}' not ready (is MAVROS running?)"
            )
            return
        self._arming_in_flight = True
        req = CommandBool.Request()
        req.value = arm
        fut = self._arming_client.call_async(req)
        fut.add_done_callback(lambda f: self._arming_done(f, arm))

    def _arming_done(self, future, requested_arm: bool) -> None:
        self._arming_in_flight = False
        try:
            resp = future.result()
            if resp.success:
                self._armed = requested_arm
                state = "ARMED" if requested_arm else "DISARMED"
                print(f"[keyboard_echo] MAVROS cmd/arming OK -> {state}", flush=True)
                self.get_logger().info(f"cmd/arming success -> {state}")
            else:
                print(
                    f"[keyboard_echo] MAVROS cmd/arming failed (success=False), "
                    f"requested arm={requested_arm}",
                    flush=True,
                )
                self.get_logger().error("cmd/arming returned success=False")
        except Exception as exc:  # noqa: BLE001
            print(f"[keyboard_echo] cmd/arming call error: {exc}", flush=True)
            self.get_logger().error(f"cmd/arming failed: {exc}")

    def _adjust_delta(self, inc: bool) -> None:
        step = 50
        if inc:
            self._delta = min(self._delta + step, 400)
        else:
            self._delta = max(self._delta - step, 50)
        pos = self._neutral + self._delta
        neg = self._neutral - self._delta
        print(
            f"[keyboard_echo] PWM span: neutral {self._neutral}, +delta -> {pos}, -delta -> {neg}",
            flush=True,
        )

    def _refresh_motion_for_char(self, ch: str) -> None:
        tags = _TTY_KEY_TO_TAGS.get(ch)
        if not tags:
            return
        now = time.monotonic()
        with self._motion_lock:
            for tag in tags:
                self._last_refresh[tag] = now

    def _is_motion_active(self, tag: str) -> bool:
        with self._motion_lock:
            t = self._last_refresh.get(tag)
        if t is None:
            return False
        return (time.monotonic() - t) <= self._idle_timeout

    def _publish_rc_override(self) -> None:
        pos = self._neutral + self._delta
        neg = self._neutral - self._delta
        nc = OverrideRCIn.CHAN_NOCHANGE
        ch = [nc] * 18
        for idx in range(6):
            ch[idx] = self._neutral

        if self._is_motion_active("up"):
            ch[2] = pos
        elif self._is_motion_active("down"):
            ch[2] = neg
        if self._is_motion_active("yaw_left"):
            ch[3] = neg
        elif self._is_motion_active("yaw_right"):
            ch[3] = pos
        if self._is_motion_active("fwd"):
            ch[4] = pos
        elif self._is_motion_active("back"):
            ch[4] = neg
        if self._is_motion_active("left"):
            ch[5] = neg
        elif self._is_motion_active("right"):
            ch[5] = pos
        if self._is_motion_active("roll_left"):
            ch[1] = neg
        elif self._is_motion_active("roll_right"):
            ch[1] = pos
        if self._is_motion_active("pitch_up"):
            ch[0] = pos
        elif self._is_motion_active("pitch_down"):
            ch[0] = neg

        msg = OverrideRCIn()
        msg.channels = ch
        self._rc_pub.publish(msg)

    def _tick(self) -> None:
        batch: list[tuple] = []
        while True:
            try:
                batch.append(self._key_q.get_nowait())
            except queue.Empty:
                break

        for item in batch:
            if item[0] == "esc":
                print("[keyboard_echo] ESC — stopping key reader", flush=True)
                self._stop_reader.set()
                rclpy.shutdown()
                return

        typed: set[str] = set()
        for item in batch:
            if item[0] == "char":
                typed.add(item[1])

        if "t" in typed and "t" not in self._prev_typed:
            self._call_arming(True)
        if "y" in typed and "y" not in self._prev_typed:
            self._call_arming(False)
        if "p" in typed and "p" not in self._prev_typed:
            self._adjust_delta(True)
        if "n" in typed and "n" not in self._prev_typed:
            self._adjust_delta(False)
        self._prev_typed = typed

        for item in batch:
            if item[0] != "char":
                continue
            ch = item[1]
            self._refresh_motion_for_char(ch)
            label = _KEY_LABELS.get(ch, "unknown")
            disp = repr(" ") if ch == " " else repr(ch)
            print(f"[keyboard_echo] key {disp} -> {label}", flush=True)

        self._publish_rc_override()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BrovKeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_reader.set()
        node._reader.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
