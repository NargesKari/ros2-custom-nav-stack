#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, TransformStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import TransformBroadcaster
import tf_transformations
import numpy as np
import math

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')

        self.num_particles = 200   
        self.particles = []          
        
        self.map_data = None
        self.resolution = 0.05
        self.width = 0
        self.height = 0
        self.origin = [0, 0]
        self.last_odom = None
        self.initialized = False

        self.alpha_rot = 0.1 
        self.alpha_trans = 0.1 

        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)
        self.cloud_pub = self.create_publisher(PoseArray, '/particlecloud', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        

        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/ekf_diff_imu/odom', self.odom_callback, odom_qos)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info("Particle Filter Node Started via ROS 2 Jazzy...")

    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.resolution = msg.info.resolution
        self.width = msg.info.width
        self.height = msg.info.height
        self.origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        if not self.initialized:
            self.initialize_particles()

    def initialize_particles(self):
        self.particles = []
        attempts = 0
        while len(self.particles) < self.num_particles and attempts < 10000:
            rx = np.random.uniform(0, self.width)
            ry = np.random.uniform(0, self.height)
            
            if self.map_data[int(ry), int(rx)] == 0:
                wx = rx * self.resolution + self.origin[0]
                wy = ry * self.resolution + self.origin[1]
                theta = np.random.uniform(-math.pi, math.pi)
                self.particles.append([wx, wy, theta, 1.0])
            attempts += 1
            
        self.initialized = True
        self.get_logger().info(f"Initialized {len(self.particles)} particles.")
        self.publish_particles()

    def odom_callback(self, msg):
        if not self.initialized: return

        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        current_odom = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])

        if self.last_odom is not None:
            dx = current_odom[0] - self.last_odom[0]
            dy = current_odom[1] - self.last_odom[1]
            dtheta = current_odom[2] - self.last_odom[2]

            dist = math.sqrt(dx**2 + dy**2)
            if dist > 0.001 or abs(dtheta) > 0.001:
                cos_a = math.cos(self.last_odom[2])
                sin_a = math.sin(self.last_odom[2])
                d_trans = dx * cos_a + dy * sin_a
                
                new_particles = []
                for p in self.particles:
                    noise_t = d_trans + np.random.normal(0, self.alpha_trans * dist)
                    noise_r = dtheta + np.random.normal(0, self.alpha_rot * abs(dtheta))
                    
                    p[2] += noise_r
                    p[0] += noise_t * math.cos(p[2])
                    p[1] += noise_t * math.sin(p[2])
                    
                    p[2] = math.atan2(math.sin(p[2]), math.cos(p[2]))
                    new_particles.append(p)
                
                self.particles = new_particles
                self.publish_particles()

        self.last_odom = current_odom

    def scan_callback(self, msg):
        if not self.initialized or self.map_data is None: return

        self.get_logger().info(f'Laser Scan received! Number of beams: {len(msg.ranges)}', throttle_duration_sec=2.0)
        step = 10
        ranges = msg.ranges[::step]
        angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment * step)

        weights = []
        total_w = 0.0

        for p in self.particles:
            score = 0.0
            valid_points = 0
            
            for r, ang in zip(ranges, angles):
                if r < msg.range_min or r > msg.range_max: continue
                
                hit_x = p[0] + r * math.cos(p[2] + ang)
                hit_y = p[1] + r * math.sin(p[2] + ang)
                
                mx = int((hit_x - self.origin[0]) / self.resolution)
                my = int((hit_y - self.origin[1]) / self.resolution)
                
                if 0 <= mx < self.width and 0 <= my < self.height:
                    if self.map_data[my, mx] > 50:
                        score += 1.0
                    valid_points += 1
            
            w = (score / (valid_points + 1)) ** 2 
            weights.append(w + 1e-10) 
            
     
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        
        n_eff = 1.0 / sum([w**2 for w in weights])
        
        if n_eff < self.num_particles / 2.0:
            
            keep_ratio = 0.90  
            num_keep = int(self.num_particles * keep_ratio)
            num_random = self.num_particles - num_keep  

            indices = np.random.choice(len(self.particles), num_keep, p=weights)
            
            new_particles = []
            for i in indices:
                p = list(self.particles[i]) 
                p[0] += np.random.normal(0, 0.05) 
                p[1] += np.random.normal(0, 0.05) 
                p[2] += np.random.normal(0, 0.05) 
                new_particles.append(p)
            
           
            attempts = 0
            injected = 0
            while injected < num_random and attempts < 2000:
                rx = np.random.uniform(0, self.width)
                ry = np.random.uniform(0, self.height)
                
            
                if self.map_data[int(ry), int(rx)] == 0: 
                    wx = rx * self.resolution + self.origin[0]
                    wy = ry * self.resolution + self.origin[1]
                    theta = np.random.uniform(-math.pi, math.pi)
                    new_particles.append([wx, wy, theta, 1.0 / self.num_particles])
                    injected += 1
                attempts += 1
            
            self.particles = new_particles
        
        self.estimate_and_publish_pose()

    def estimate_and_publish_pose(self):
        x_avg = np.mean([p[0] for p in self.particles])
        y_avg = np.mean([p[1] for p in self.particles])
        
        sin_avg = np.mean([math.sin(p[2]) for p in self.particles])
        cos_avg = np.mean([math.cos(p[2]) for p in self.particles])
        yaw_avg = math.atan2(sin_avg, cos_avg)
        
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = x_avg
        msg.pose.pose.position.y = y_avg
        q = tf_transformations.quaternion_from_euler(0, 0, yaw_avg)
        msg.pose.pose.orientation.x = q[0]
        msg.pose.pose.orientation.y = q[1]
        msg.pose.pose.orientation.z = q[2]
        msg.pose.pose.orientation.w = q[3]
        self.pose_pub.publish(msg)
        
   
    def publish_particles(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for p in self.particles:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            q = tf_transformations.quaternion_from_euler(0, 0, p[2])
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)
        self.cloud_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()