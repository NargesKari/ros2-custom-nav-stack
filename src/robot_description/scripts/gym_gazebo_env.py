
from __future__ import annotations

import time
import threading
import subprocess
import shutil
import os
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldControl, WorldReset

# -----------------------------------------------------------------------------
# Tunable constants
# -----------------------------------------------------------------------------
_SERVICE_TIMEOUT = 10.0        # seconds waiting for the ControlWorld service call
_OBS_WAIT_TIMEOUT = 2.0        # seconds to wait for a fresh odometry update
_SPAWN_WAIT_TIMEOUT = 6.0      # seconds to wait for odometry after spawning the robot


# -----------------------------------------------------------------------------
# Helper class: single rclpy node with background spinning
# -----------------------------------------------------------------------------
class _RosNodeHolder:

    def __init__(self, node_name: str = 'gym_gazebo_node'):
        # Initialize rclpy once per process if not already initialized.
        if not rclpy.ok():
            rclpy.init(args=None)

        # Create the ROS node used for publishers/subscriptions/service clients.
        self.node: Node = Node(node_name)

        # Start a background thread that runs rclpy.spin(node).
        self._executor_thread = threading.Thread(target=self._spin, daemon=True)
        self._executor_thread.start()

    def _spin(self) -> None:
        
        try:
            rclpy.spin(self.node)
        except Exception:
            pass

    def shutdown(self) -> None:

        try:
            self.node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

class GazeboEnv(gym.Env):
   
    metadata = {"render.modes": []}

    def __init__(
        self,
        world_name: str = 'depot',
        cmd_topic: str = '/cmd_vel',
        odom_topic: str = '/wheel_encoder/odom',
        sim_steps_per_env_step: int = 1,
        spawn_name: str = 'robot',
        spawn_pose: Tuple[float, float, float] = (0.0, 0.0, 0.9)
    ):
        super().__init__()

        self._ros = _RosNodeHolder(node_name='gym_gazebo_env_node')

        self._world = world_name
        self._control_service_name = f'/world/{self._world}/control'
        self._sim_steps_per_env_step = int(sim_steps_per_env_step)

        self._spawn_name = spawn_name
        self._spawn_pose = spawn_pose

        self._cmd_pub = self._ros.node.create_publisher(Twist, cmd_topic, 10)

        qos_odom = QoSProfile(depth=10)
        qos_odom.reliability = QoSReliabilityPolicy.RELIABLE
        qos_odom.history = QoSHistoryPolicy.KEEP_LAST

        self._last_odom: Optional[np.ndarray] = None
        self._odom_lock = threading.Lock()
        self._odom_timestamp: float = 0.0

        self._odom_sub = self._ros.node.create_subscription(
            Odometry, odom_topic, self._odom_cb, qos_odom
        )

        self._control_client = self._ros.node.create_client(ControlWorld, self._control_service_name)
        if not self._control_client.wait_for_service(timeout_sec=_SERVICE_TIMEOUT):
            raise RuntimeError(
                f"Timeout waiting for service {self._control_service_name}. "
                "Ensure ros_gz_bridge is running and bridging the ControlWorld service."
            )

        self.action_space = spaces.Box(low=np.array([-1.0, -3.14]), high=np.array([1.0, 3.14]), dtype=np.float32)
        obs_high = np.array([np.finfo(np.float32).max] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Last-observation fallback
        self._last_obs = np.zeros(5, dtype=np.float32)

    def _odom_cb(self, msg: Odometry) -> None:
        with self._odom_lock:
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            # minimal quaternion -> yaw conversion (sufficient for planar robots)
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = float(np.arctan2(siny_cosp, cosy_cosp))

            vx = msg.twist.twist.linear.x
            vy = msg.twist.twist.angular.z

            self._last_odom = np.array([px, py, yaw, vx, vy], dtype=np.float32)
            self._odom_timestamp = time.time()

    def _publish_action(self, action: np.ndarray) -> None:
        t = Twist()
        t.linear.x = float(action[0])
        t.angular.z = float(action[1])
        self._cmd_pub.publish(t)

    def _call_world_control(
        self,
        *,
        pause: bool = False,
        step: bool = False,
        multi_step: int = 0,
        reset_all: bool = False,
        timeout: float = 5.0
    ):
 
        req = ControlWorld.Request()
        wc = WorldControl()
        wc.pause = bool(pause)
        wc.step = bool(step)
        if multi_step:
            wc.multi_step = int(multi_step)
        if reset_all:
            wr = WorldReset()
            wr.all = True
            wc.reset = wr
        req.world_control = wc

        future = self._control_client.call_async(req)
        t0 = time.time()
        
        while rclpy.ok() and not future.done() and (time.time() - t0) < timeout:
            time.sleep(0.001)

        if not future.done():
            raise RuntimeError("ControlWorld service call timed out.")
        return future.result()

    def _wait_for_obs_update(self, timeout: float = _OBS_WAIT_TIMEOUT) -> Optional[np.ndarray]:

        t0 = time.time()
        initial_ts = self._odom_timestamp
        while (time.time() - t0) < timeout:
            with self._odom_lock:
                if self._odom_timestamp != initial_ts and self._last_odom is not None:
                    return self._last_odom.copy()
            time.sleep(0.001)

        with self._odom_lock:
            return None if self._last_odom is None else self._last_odom.copy()

    def _spawn_robot_cli(self, name: Optional[str] = None, pose: Optional[Tuple[float, float, float]] = None) -> bool:
    
        name = name or self._spawn_name
        pose = pose or self._spawn_pose

        ros2_bin = shutil.which("ros2")
        if ros2_bin is None:
            print("[GazeboEnv] spawn failed: 'ros2' not found on PATH")
            return False

        x, y, z = pose
        cmd = [
            ros2_bin, "run", "ros_gz_sim", "create",
            "-name", name,
            "-topic", "/robot_description",
            "-x", str(x),
            "-y", str(y),
            "-z", str(z)
        ]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=os.environ, timeout=10.0)
            if proc.returncode == 0:
                print(f"[GazeboEnv] spawn CLI succeeded (name={name}).")
                return True
            else:
                stderr = proc.stderr.decode('utf-8', errors='ignore')
                print(f"[GazeboEnv] spawn CLI failed, return code {proc.returncode}. stderr:\n{stderr}")
                return False
        except Exception as e:
            print(f"[GazeboEnv] spawn CLI exception: {e}")
            return False

    def step(self, action):

        action = np.asarray(action, dtype=np.float32)
        assert self.action_space.contains(action), f"Action out of bounds: {action}"

        self._publish_action(action)

        self._call_world_control(
            pause=True,
            multi_step=self._sim_steps_per_env_step,
            timeout=_SERVICE_TIMEOUT
        )

        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            obs = self._last_obs.copy()
        else:
            self._last_obs = obs

        reward = -np.linalg.norm(obs[:2])
        terminated = False
        truncated = False
        info = {}

        return obs, float(reward), bool(terminated), bool(truncated), info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        try:
            self._call_world_control(reset_all=True, timeout=_SERVICE_TIMEOUT)
        except Exception as e:
            print(f"[GazeboEnv] world reset call failed: {e}")

        time.sleep(0.2)

        with self._odom_lock:
            self._last_odom = None
            self._odom_timestamp = 0.0

        spawned = self._spawn_robot_cli(name=self._spawn_name, pose=self._spawn_pose)
        if spawned:
            obs = self._wait_for_obs_update(timeout=_SPAWN_WAIT_TIMEOUT)
            if obs is not None:
                self._last_obs = obs
                return obs, {}
            else:
                print("[GazeboEnv] spawn completed but no odom received within timeout.")
        else:
            print("[GazeboEnv] spawn attempt failed. Robot may not be present.")

        obs = self._wait_for_obs_update(timeout=_OBS_WAIT_TIMEOUT)
        if obs is None:
            obs = np.zeros(5, dtype=np.float32)
        self._last_obs = obs
        return obs, {}

    def close(self) -> None:
        try:
            self._ros.shutdown()
        except Exception:
            pass
