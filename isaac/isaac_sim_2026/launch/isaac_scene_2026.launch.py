#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    assets_root = "/home/u-zhuang/ros2_ws/src/isaac/isaac-sim/isaacsimassets-dev"
    run_script = f"{assets_root}/integrated_runtime/run_demo_scene.sh"

    world = LaunchConfiguration("world")
    isaacsim_root = LaunchConfiguration("isaacsim_root")
    isaacsim_launcher = LaunchConfiguration("isaacsim_launcher")
    pegasus_extension = LaunchConfiguration("pegasus_extension")
    code_editor_port = LaunchConfiguration("code_editor_port")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="X1",
                description="Official 2026 scene: X1, Bobac, or create_map",
            ),
            DeclareLaunchArgument(
                "isaacsim_root",
                default_value="/home/u-zhuang/isaac-sim-4.5.0",
                description="Local Isaac Sim 4.5 root",
            ),
            DeclareLaunchArgument(
                "isaacsim_launcher",
                default_value="/home/u-zhuang/isaac-sim-4.5.0/isaac-sim.sh",
                description="Isaac Sim launcher script",
            ),
            DeclareLaunchArgument(
                "pegasus_extension",
                default_value=(
                    "/home/u-zhuang/ros2_ws/src/robotac_airobotic_uav_project/"
                    "reference_projects/PegasusSimulator/extensions/pegasus.simulator"
                ),
                description="Pegasus simulator extension path",
            ),
            DeclareLaunchArgument(
                "code_editor_port",
                default_value="8226",
                description="isaacsim.code_editor.vscode TCP control port",
            ),
            ExecuteProcess(
                cmd=[
                    "bash",
                    run_script,
                    "--world",
                    world,
                    "--enable",
                    "isaacsim.code_editor.vscode",
                    "--/exts/isaacsim.code_editor.vscode/host=127.0.0.1",
                    [
                        "--/exts/isaacsim.code_editor.vscode/port=",
                        code_editor_port,
                    ],
                ],
                cwd=f"{assets_root}/integrated_runtime",
                additional_env={
                    "ISAACSIM_ROOT": isaacsim_root,
                    "ISAACSIM_LAUNCHER": isaacsim_launcher,
                    "PEGASUS_EXTENSION": pegasus_extension,
                },
                output="screen",
            ),
        ]
    )
