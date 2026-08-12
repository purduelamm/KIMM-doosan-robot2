"""Run one timed air-blow trigger test using the blow-from-mesh tool control."""

import time

import rclpy

from blow_from_mesh_helpers import ros_context


BLOW_DURATION_SEC = 10.0
WAIT_DURATION_SEC = 10.0


def main(args=None):
    ros_context.init_ros()
    airgun_enabled = False

    try:
        print(f"[blow-test] Blowing for {BLOW_DURATION_SEC:g} seconds...")
        ros_context.air_node.tool_airgun(True)
        airgun_enabled = True
        time.sleep(BLOW_DURATION_SEC)

        ros_context.air_node.tool_airgun(False)
        airgun_enabled = False
        print(f"[blow-test] Blow stopped. Waiting for {WAIT_DURATION_SEC:g} seconds...")
        time.sleep(WAIT_DURATION_SEC)
        print("[blow-test] Test complete.")
    finally:
        # Ensure the output is turned off if the test is interrupted or fails.
        if airgun_enabled:
            ros_context.air_node.tool_airgun(False)
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
