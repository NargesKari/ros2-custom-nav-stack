# ros2-custom-nav-stack

A ROS 2 navigation stack built from scratch instead of `nav2`: a custom occupancy-grid SLAM node, a Monte Carlo particle-filter localizer, an EKF sensor-fusion node, and an A* global planner, all wired together and verified against a Gazebo simulation.

## What it does

- **`map_publisher.py`** — reads a `.pgm`/`.yaml` map pair from disk, converts pixel intensities to ROS occupancy probabilities (0 = free, 100 = occupied, -1 = unknown), and republishes it on `/map` with `TRANSIENT_LOCAL` durability so RViz or any node that starts late still gets the latest map.
- **`custom_slam.py`** — builds an occupancy grid live from `/scan` and odometry, using Bresenham line tracing to mark free cells along each laser ray and the endpoint as occupied, and exposes a `/save_map` service to persist the map to disk.
- **`particle_filter.py`** — a from-scratch Monte Carlo Localization implementation: 200 particles scattered only in known-free cells, motion-model prediction from odometry with proportional Gaussian noise, raycast-based sensor-weight scoring, and a hybrid resampling step (90% weighted resampling + 10% random re-injection) specifically to recover from the kidnapped-robot problem instead of converging permanently on a wrong pose.
- **`ekf_diff_imu.cpp`** — an Eigen-based EKF that fuses wheel-encoder odometry with IMU angular velocity into a single `[x, y, theta]` state estimate and broadcasts the corresponding TF transform.
- **`astar.cpp`** — a grid-based A* global planner that waits for a map, an AMCL-style pose, and a goal, then publishes a `nav_msgs/Path`.
- **`frame_id_converter.cpp`** and **`motor_command.cpp`** — small integration nodes: the first patches an incorrect `frame_id` on Gazebo's simulated LiDAR point cloud, the second fans a two-element motor-command array out to separate left/right RPM topics.
- **`gym_gazebo_env.py`** — a Gymnasium `Env` wrapper around the same Gazebo world, driving the simulation deterministically through Gazebo's `ControlWorld` service (pause/step/multi-step) instead of free-running wall-clock time, so an RL agent can be trained against this robot and world.

## Why it's interesting

Most of this exists specifically because `nav2`'s built-in AMCL and planner were deliberately not used — the particle filter, the EKF, and the A* planner are all hand-implemented against raw topics (`/scan`, `/map`, wheel odometry, IMU), which means every one of the usual pitfalls had to be handled explicitly rather than inherited from a maintained package. Two examples: the particle filter's resampling step doesn't purely exploit high-weight particles, because doing so causes permanent convergence on a wrong pose the moment early observations are misleading — the 10%-random-injection floor is what lets it recover. And the SLAM node's use of Bresenham line tracing (rather than just marking scan endpoints as occupied) is what actually clears previously-unknown cells to free space as the robot passes through them, which is the difference between "a scatter plot of hits" and an actual occupancy map a planner can use.

The Gazebo RL environment is the other non-obvious piece: naively stepping a Gym environment against a live Gazebo sim races the physics engine against the RL loop's wall-clock timing. This wrapper instead calls Gazebo's `ControlWorld` service to explicitly pause the world and step it a fixed number of physics ticks per `env.step()`, making training reproducible regardless of how fast the training loop itself runs on a given machine.

## Tech stack

ROS 2, C++ (EKF, A*, and two integration nodes, built with `ament_cmake` and Eigen3), Python (map publishing, SLAM, particle filter, installed via `ament_cmake_python`), Gazebo (via `ros_gz_bridge`/`ros_gz_interfaces`) for simulation, `gymnasium` for the RL environment wrapper, RViz for visualization.

## Getting started

Requires a ROS 2 workspace with Gazebo and the `ros_gz` bridge packages installed.

```bash
cd src/robot_description/..   # workspace root
colcon build
source install/setup.bash
ros2 launch robot_description gazebo.launch.py   # spawns the robot in the depot world
ros2 launch robot_description display.launch.py  # brings up map_publisher + RViz
```

Run the custom stack's nodes individually once the simulation is up:

```bash
ros2 run robot_description ekf_diff_imu_node
ros2 run robot_description astar_node
ros2 run robot_description particle_filter.py
ros2 run robot_description custom_slam.py
```

To build your own map instead of using the bundled `maps/depot.yaml`: run `custom_slam.py` while driving the robot around, then `ros2 service call /save_map std_srvs/srv/Empty`.

## Architecture

The pipeline is `map_publisher` (or `custom_slam` while mapping) → `particle_filter` (publishes `/amcl_pose` and the `map`→`odom` TF) → `astar_node` (consumes the map, the localized pose, and `/goal_pose` to publish `/global_path`), with `ekf_diff_imu_node` running in parallel to provide the fused odometry that both the particle filter and SLAM node consume. `gym_gazebo_env.py` sits outside this pipeline as an alternative RL-facing interface to the same simulated robot.

<!-- add screenshot/demo here -->
