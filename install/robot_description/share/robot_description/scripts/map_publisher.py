#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from ament_index_python.packages import get_package_share_directory
import yaml
import numpy as np
from PIL import Image
import os

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher_ = self.create_publisher(OccupancyGrid, '/map', qos)

        package_name = 'robot_description' 
        share_dir = get_package_share_directory(package_name)
        
        self.declare_parameter('yaml_path', os.path.join(share_dir, 'maps', 'depot.yaml'))
        yaml_file = self.get_parameter('yaml_path').get_parameter_value().string_value
        
        self.get_logger().info(f'Loading map from: {yaml_file}')
        try:
            self.load_and_publish(yaml_file)
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')

    def load_and_publish(self, yaml_file):
        with open(yaml_file, 'r') as f:
            map_config = yaml.safe_load(f)

        map_dir = os.path.dirname(yaml_file)
        img_path = os.path.join(map_dir, map_config['image'])
        
        img = Image.open(img_path)
        img_data = np.flipud(np.array(img.convert('L')))
        
        p = img_data / 255.0
        if map_config['negate'] == 0:
            p = 1.0 - p
            
        map_data = np.full(p.shape, -1, dtype=np.int8)
        map_data[p >= map_config['occupied_thresh']] = 100
        map_data[p <= map_config['free_thresh']] = 0
        
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = map_config['resolution']
        msg.info.width = map_data.shape[1]
        msg.info.height = map_data.shape[0]
        msg.info.origin.position.x = float(map_config['origin'][0])
        msg.info.origin.position.y = float(map_config['origin'][1])
        msg.info.origin.orientation.w = 1.0
        msg.data = map_data.flatten().tolist()
        
        self.publisher_.publish(msg)
        self.get_logger().info('Map Published!')

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

         



