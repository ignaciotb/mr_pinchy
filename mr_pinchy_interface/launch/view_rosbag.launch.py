"""RViz visualization for a recorded rosbag.

Brings up RViz plus the pieces the bag does not carry: the bag only records TF
from `mocap` down (mocap -> mr_pinchy/base_link[_ekf] and the static sensor
frames), while the RViz configs use `map` as fixed frame, so the missing static
frames and the URDF descriptions are published here.

By default it does not touch the bag - play it yourself in a second terminal so
you keep the player's keyboard controls (SPACE pause/resume, -> step one
message, up/down rate +-10%):

    ros2 launch mr_pinchy_interface view_rosbag.launch.py
    ros2 bag play ~/workspace/underwater_ws/data_rosbag/rosbag2_2026_06_20-17_32_04_cut --clock --loop

Play with `--clock`, since everything here defaults to use_sim_time:=true. If
you would rather play without it, launch with use_sim_time:=false.

The bag's robot marker is recorded blue; it is republished yellow here, so play
the bag with the printed --remap flag (or pass recolor_robot_marker:=false to
see the original colour). Pick another colour with, for example,
robot_marker_color:="0.2,1.0,0.2".

To have the launch file play the bag instead (rate/loop/start_offset/topics
args then apply, but the keyboard controls are dead because the player gets no
terminal - use `ros2 service call /rosbag2_player/toggle_paused
rosbag2_interfaces/srv/TogglePaused` and friends):

    ros2 launch mr_pinchy_interface view_rosbag.launch.py play:=true loop:=true rate:=0.5
"""

import math
import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# The recorded /safety_filter/robot_marker is baked blue and RViz cannot
# override a marker colour, so the bag's copy is remapped aside and
# marker_recolor republishes it under the original name in `robot_marker_color`.
ROBOT_MARKER_TOPIC = "/safety_filter/robot_marker"
ROBOT_MARKER_RAW_TOPIC = ROBOT_MARKER_TOPIC + "_raw"

DEFAULT_BAG = (
    "/home/shaohang/workspace/underwater_ws/data_rosbag/"
    "rosbag2_2026_06_20-17_32_04_cut"
)


def _is_true(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def _parse_rgba(value: str) -> tuple:
    """Parse "r,g,b" or "r,g,b,a"; a negative alpha keeps the incoming one."""
    parts = [float(v) for v in value.split(",")]
    if len(parts) == 3:
        parts.append(-1.0)
    if len(parts) != 4:
        raise RuntimeError(
            f"expected a colour as 'r,g,b' or 'r,g,b,a', got '{value}'"
        )
    return tuple(parts)


def _bag_play_action(context, *args, **kwargs) -> list:
    """Build `ros2 bag play` with only the flags the user actually asked for."""
    bag = LaunchConfiguration("bag").perform(context)
    rate = LaunchConfiguration("rate").perform(context)
    start_offset = LaunchConfiguration("start_offset").perform(context)
    topics = LaunchConfiguration("topics").perform(context).strip()
    play_delay = float(LaunchConfiguration("play_delay").perform(context))

    # --clock feeds /clock so every node below can run on bag time.
    cmd = ["ros2", "bag", "play", bag, "--clock", "--rate", rate]
    if float(start_offset) > 0.0:
        cmd += ["--start-offset", start_offset]
    if _is_true(LaunchConfiguration("loop").perform(context)):
        cmd.append("--loop")
    if _is_true(LaunchConfiguration("start_paused").perform(context)):
        cmd.append("--start-paused")
    if topics:
        cmd += ["--topics"] + topics.split()

    actions = []
    if _is_true(LaunchConfiguration("recolor_robot_marker").perform(context)):
        r, g, b, a = _parse_rgba(
            LaunchConfiguration("robot_marker_color").perform(context)
        )
        cmd += ["--remap", f"{ROBOT_MARKER_TOPIC}:={ROBOT_MARKER_RAW_TOPIC}"]
        actions.append(
            Node(
                package="mr_pinchy_interface",
                executable="marker_recolor",
                name="robot_marker_recolor",
                output="screen",
                parameters=[
                    {
                        "input_topic": ROBOT_MARKER_RAW_TOPIC,
                        "output_topic": ROBOT_MARKER_TOPIC,
                        "r": r, "g": g, "b": b, "alpha": a,
                    },
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            )
        )

    if not _is_true(LaunchConfiguration("play").perform(context)):
        # Let the user run the player in their own terminal, where SPACE
        # (pause/resume) and the arrow keys work.
        return actions + [
            LogInfo(msg="Play the bag in another terminal with:\n  " + " ".join(cmd))
        ]

    play = ExecuteProcess(cmd=cmd, output="screen", emulate_tty=True)

    # Give RViz and the state publishers a moment to subscribe, otherwise the
    # first messages of the bag (tf_static included) are published into the void.
    return actions + [TimerAction(period=play_delay, actions=[play])]


def generate_launch_description() -> LaunchDescription:
    default_rviz = PathJoinSubstitution(
        [FindPackageShare("mr_pinchy_interface"), "rviz", "watertank.rviz"]
    )

    bag_arg = DeclareLaunchArgument(
        "bag",
        default_value=DEFAULT_BAG,
        description="Path to the rosbag2 directory to replay",
    )
    play_arg = DeclareLaunchArgument(
        "play",
        default_value="false",
        description=(
            "Also play the bag from this launch file. Off by default: playing it "
            "yourself in a terminal keeps the player's keyboard controls"
        ),
    )
    rate_arg = DeclareLaunchArgument(
        "rate", default_value="1.0", description="Playback rate multiplier"
    )
    loop_arg = DeclareLaunchArgument(
        "loop", default_value="false", description="Restart the bag when it ends"
    )
    start_paused_arg = DeclareLaunchArgument(
        "start_paused",
        default_value="false",
        description="Start playback paused (resume via /rosbag2_player/resume)",
    )
    start_offset_arg = DeclareLaunchArgument(
        "start_offset",
        default_value="0.0",
        description="Seconds into the bag to start playing from",
    )
    topics_arg = DeclareLaunchArgument(
        "topics",
        default_value="",
        description="Optional space-separated topic whitelist (default: all)",
    )
    play_delay_arg = DeclareLaunchArgument(
        "play_delay",
        default_value="2.0",
        description="Seconds to wait before starting playback, so RViz is subscribed",
    )
    recolor_robot_marker_arg = DeclareLaunchArgument(
        "recolor_robot_marker",
        default_value="true",
        description=(
            f"Remap the bag's {ROBOT_MARKER_TOPIC} aside and republish it in "
            "robot_marker_color, since the recorded colour cannot be overridden"
        ),
    )
    robot_marker_color_arg = DeclareLaunchArgument(
        "robot_marker_color",
        default_value="1.0,0.85,0.1",
        description=(
            "Robot marker colour as 'r,g,b' or 'r,g,b,a' in 0-1 "
            "(default yellow, keeping the recorded alpha)"
        ),
    )
    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=default_rviz,
        description="Absolute path to RViz config file",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use the bag clock (keep true while replaying)",
    )
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    # map -> mocap: identity translation, RPY [-180, 0, 90] deg. Same convention
    # as overlay_rviz.launch.py; the bag's TF tree is rooted at `mocap`.
    map_to_mocap = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_mocap",
        arguments=[
            "--frame-id", "map", "--child-frame-id", "mocap",
            "--roll", str(-math.pi), "--pitch", "0.0", "--yaw", str(math.pi / 2),
        ],
        parameters=[use_sim_time],
    )

    # The watertank URDF is a single free-floating link, so nothing publishes a
    # transform for it. Pin it to `map` so the tank mesh shows up in RViz.
    map_to_tank = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_tank_base",
        arguments=["--frame-id", "map", "--child-frame-id", "tank_base"],
        parameters=[use_sim_time],
    )

    watertank_description = os.path.join(
        get_package_share_directory("mr_pinchy_interface"),
        "urdf", "watertank_description", "robots", "watertank_default.urdf.xacro",
    )
    watertank_description = xacro.process_file(watertank_description).toxml()
    watertank_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="watertank_state_publisher",
        parameters=[{"robot_description": watertank_description}, use_sim_time],
        remappings=[("robot_description", "tank_description")],
    )

    # Publishes /robot_description so a RobotModel display can show the ROV on
    # the mr_pinchy/base_link_ekf_flu frame that the bag provides.
    robot_description = os.path.join(
        get_package_share_directory("mr_pinchy_interface"),
        "urdf", "bluerov2_heavy.urdf.xacro",
    )
    robot_description = xacro.process_file(robot_description).toxml()
    robot_state_pub = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": robot_description}, use_sim_time],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[use_sim_time],
    )

    return LaunchDescription(
        [
            bag_arg,
            play_arg,
            rate_arg,
            loop_arg,
            start_paused_arg,
            start_offset_arg,
            topics_arg,
            play_delay_arg,
            recolor_robot_marker_arg,
            robot_marker_color_arg,
            rviz_config_arg,
            use_sim_time_arg,
            map_to_mocap,
            map_to_tank,
            watertank_state_pub,
            robot_state_pub,
            rviz_node,
            OpaqueFunction(function=_bag_play_action),
        ]
    )
