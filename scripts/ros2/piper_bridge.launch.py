from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="gr00t_piper_bridge",
                executable="piper_bridge",
                name="gr00t_piper_bridge",
                output="screen",
            )
        ]
    )