# 2026 Stage 1 Adaptation

This package is a 2026-safe copy of the previous `isaac_sim` project. The ROS
package name is changed to `isaac_sim_2026` so it can coexist with the old
package in the same workspace.

## Official Assets

Use the 2026 official scene from:

```bash
/home/u-zhuang/ros2_ws/src/isaac/isaac-sim/isaacsimassets-dev
```

Start the X1 official scene first:

```bash
ros2 launch isaac_sim_2026 isaac_scene_2026.launch.py world:=X1
```

For map creation or inspection, use:

```bash
ros2 launch isaac_sim_2026 isaac_scene_2026.launch.py world:=create_map
ros2 launch isaac_sim_2026 slam_2026.launch.py
```

The official runtime publishes `/avoidance/lidar/pointcloud`,
`/drone_0_ego_odom`, `/cargo_bay/command`, `/cargo_bay/status`,
`/drone/world_position`, `/clock`, and TF. This package adds
`pointcloud_to_laserscan_2026.py`, which converts the official 3D point cloud to
`/laser_scan_fuse` and `/scan` for SLAM Toolbox, AMCL, and Nav2.

## Build

```bash
cd /home/u-zhuang/ros2_ws
colcon build --symlink-install \
  --base-paths \
    /home/u-zhuang/ros2_ws/src/isaac/isaac-sim/demo_ws/src \
    /home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026 \
  --packages-select grasp_demo_interfaces grasp_demo_pkg nav2_demo_pkg isaac_sim_2026
source install/setup.bash
```

## Stage 1 Entry Point

```bash
ros2 launch isaac_sim_2026 stage1_2026.launch.py
```

The launch starts the official grasp demo perception/action nodes and the new
`stage1_task_manager.py` sequence:

1. send a Nav2 goal to the office/material area;
2. run YOLOE material detection without AprilTag/manual visual labels;
3. wait for depth + TF object pose estimation;
4. command pre-grasp, grasp, lift, cargo-place and gripper actions;
5. publish cargo-bay open/close commands on `/cargo_bay/command`.

The manager also listens to `/cargo_bay/status` and warns if the expected
open/close confirmation is not received.

Nav2 is disabled by default in this launch because many setups already start it
separately. To start the included official Nav2 demo nodes:

```bash
ros2 launch isaac_sim_2026 navigation2_2026.launch.py
```

or let the stage-1 launch start the same Nav2 stack:

```bash
ros2 launch isaac_sim_2026 stage1_2026.launch.py start_nav:=true
```

`navigation2_2026.launch.py`, `slam_2026.launch.py`, and `stage1_2026.launch.py`
load `nav2_2026_overrides.yaml`, which maps Nav2/SLAM to the official 2026
frames:

- base frame: `avoidance_base_link`
- lidar frame: `avoidance_lidar`
- odometry: `/drone_0_ego_odom`
- laser scan compatibility topic: `/laser_scan_fuse`

## Parameters To Tune On The 2026 Field

Edit:

```text
/home/u-zhuang/ros2_ws/src/isaac/isaac_sim_2026/config/stage1_2026.yaml
```

Key fields:

- `office_goal`: `[x, y, yaw]` in the map frame. It is currently a placeholder
  and must be set after building/localizing against the 2026 map.
- `cargo_place_position`: target point for placing the material into the
  drone cargo bay, in the grasp demo base frame.
- `target_classes`: default `pencil, pen`, matching the official demo objects.
- `execute_demo_motion`: uses the official teaching `PlanToPose` action. Replace
  this with the copied cuRobo executor for scored precise grasping if needed.

The technical compliance path is perception-based YOLOE + depth pose estimation;
the copied AprilTag tools are preserved for reference/debug only and should not
be used in the scored run.
