#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import heapq
import math

class AStarInteractive(Node):
    def __init__(self):
        super().__init__('a_star_interactive')

        self.map_data = None
        self.resolution = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.width = 0
        self.height = 0
        
        self.start_pose = None  # (x, y)
        self.goal_pose = None   # (x, y)


        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, map_qos)


        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_pose_callback, 10)

        self.path_pub = self.create_publisher(Path, '/plan', 10)

        self.get_logger().info("A* Interactive Node Started!")
        self.get_logger().info("1. Set start point using '2D Pose Estimate' in RViz.")
        self.get_logger().info("2. Set goal point using '2D Goal Pose' in RViz.")

    def map_callback(self, msg):
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        self.map_data = np.array(msg.data).reshape((self.height, self.width))
        self.get_logger().info("Map loaded successfully!")

    def initial_pose_callback(self, msg):
        self.start_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().info(f"Start Point Set: X={self.start_pose[0]:.2f}, Y={self.start_pose[1]:.2f}")

    def goal_pose_callback(self, msg):
        self.goal_pose = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"Goal Point Set: X={self.goal_pose[0]:.2f}, Y={self.goal_pose[1]:.2f}")
        
        self.plan_path()

    def world_to_grid(self, wx, wy):
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return (mx, my)

    def grid_to_world(self, mx, my):
        wx = (mx * self.resolution) + self.origin_x + (self.resolution / 2.0)
        wy = (my * self.resolution) + self.origin_y + (self.resolution / 2.0)
        return (wx, wy)

    def is_valid(self, mx, my):
        if mx < 0 or mx >= self.width or my < 0 or my >= self.height:
            return False
        val = self.map_data[my, mx]
        if val > 50 or val == -1:
            return False
        return True

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(self, node):
        directions = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0), 
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]
        neighbors = []
        for dx, dy, cost in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if self.is_valid(nx, ny):
                neighbors.append(((nx, ny), cost))
        return neighbors

    def plan_path(self):
        if self.map_data is None:
            self.get_logger().warn("Waiting for map...")
            return
        if self.start_pose is None:
            self.get_logger().warn("Please set Start Point first using '2D Pose Estimate'!")
            return

        start_grid = self.world_to_grid(self.start_pose[0], self.start_pose[1])
        goal_grid = self.world_to_grid(self.goal_pose[0], self.goal_pose[1])

        if not self.is_valid(start_grid[0], start_grid[1]):
            self.get_logger().error("Start point is inside an obstacle!")
            return
        if not self.is_valid(goal_grid[0], goal_grid[1]):
            self.get_logger().error("Goal point is inside an obstacle!")
            return

        self.get_logger().info("Calculating A* Path...")
        path_grid = self.a_star(start_grid, goal_grid)

        if path_grid:
            self.get_logger().info(f"Path found! Length: {len(path_grid)} nodes.")
            self.publish_path(path_grid)
        else:
            self.get_logger().error("No path found! Target might be blocked.")

    def a_star(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor, cost in self.get_neighbors(current):
                tentative_g_score = g_score[current] + cost

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def publish_path(self, path_grid):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for mx, my in path_grid:
            wx, wy = self.grid_to_world(mx, my)
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0 
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AStarInteractive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()