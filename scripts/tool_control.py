# tool_control.py
import rclpy
from dsr_msgs2.srv import SetToolDigitalOutput

class ToolControlNode:
    def __init__(self, node):
        # Use the existing node instead of creating a new one
        self.node = node
        self.client = self.node.create_client(
            SetToolDigitalOutput, 
            'io/set_tool_digital_output'  # Relative path - will use node's namespace
        )
        
        # Wait for service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Service not available, waiting again...')
        self.node.get_logger().info('Tool digital output service is ready')

    def tool_airgun(self, state: bool):
        request = SetToolDigitalOutput.Request()
        request.index = 6
        request.value = state
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        
        if future.result() is not None:
            if state:
                self.node.get_logger().info("Airgun is activated")
            else:
                self.node.get_logger().info("Airgun is deactivated")
        else:
            self.node.get_logger().error("Failed to call service")