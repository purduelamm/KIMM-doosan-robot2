import rclpy
import time
from tool_control import ToolControlNode

# First Run this.
# ros2 launch dsr_bringup2 dsr_bringup2_moveit.launch.py mode:=real model:=m0609 host:=192.168.137.100


if __name__ == '__main__':
    rclpy.init()
    # Create node
    node = ToolControlNode()
    
    while True:
        try:
            node.tool_airgun(True)
            time.sleep(1)
            node.tool_airgun(False)
            time.sleep(1)
        except KeyboardInterrupt:
            node.tool_airgun(False)
            time.sleep(1)
            # Clean up and shut down ROS2
            rclpy.shutdown()