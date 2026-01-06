import rclpy
import DR_init
import sys
import time
from tool_control import ToolControlNode

def main(args=None):
    rclpy.init(args=args)
    ROBOT_ID = "dsr01"
    ROBOT_MODEL = "m0609"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL
    node = rclpy.create_node('coverage_path', namespace=ROBOT_ID)
    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, posj, movel, posx, set_robot_mode, get_current_posx, ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    air_node = ToolControlNode(node)

    time.sleep(5)
    air_node.tool_airgun(True)
    time.sleep(1)
    # Move to home position
    home_pos = posj(90, 0, -90, 0, 0, 0)
    movej(home_pos, vel=30, acc=30)
    time.sleep(1)
    air_node.tool_airgun(False)
    time.sleep(1)
    home_pos = posj(-70, 0, -90, 0, 0, 0)
    movej(home_pos, vel=30, acc=30)
    time.sleep(1)
    home_pos = posj(-70, -30, -60, 0, 0, 0)
    movej(home_pos, vel=30, acc=30)
    time.sleep(1)
    home_pos = posj(-70, -30, -60, 0, -70, 0)
    movej(home_pos, vel=30, acc=30)
    time.sleep(1)

    # Coverage area parameters (mm)
    box_width = 250.0   # X direction
    box_height = 170.0  # Y direction
    line_spacing = 10.0 # Distance between parallel lines
    
    # Motion parameters
    vel = 30
    acc = 30

    # Get home position as center point
    center = get_current_posx()[0]
    print(f"Center position: {center}")

    # Store orientation (keep constant)
    rx, ry, rz = center[3], center[4], center[5]
    
    # Calculate starting corner (bottom-left) from center
    start_x = center[0] - box_width / 2
    start_y = center[1] - box_height / 2
    start_z = center[2]

    # Generate coverage path (zigzag pattern) centered on home
    num_lines = int(box_height / line_spacing) + 1
    
    path = []
    for i in range(num_lines):
        y_offset = i * line_spacing
        
        if i % 2 == 0:
            # Even lines: left to right
            path.append(posx(start_x, start_y + y_offset, start_z, rx, ry, rz))
            path.append(posx(start_x + box_width, start_y + y_offset, start_z, rx, ry, rz))
        else:
            # Odd lines: right to left
            path.append(posx(start_x + box_width, start_y + y_offset, start_z, rx, ry, rz))
            path.append(posx(start_x, start_y + y_offset, start_z, rx, ry, rz))

    # Move to first waypoint
    print(f"Coverage path: {box_width}x{box_height}mm area centered on home")
    print(f"Line spacing: {line_spacing}mm, Total lines: {num_lines}")
    print(f"Box bounds: X[{start_x:.1f}, {start_x + box_width:.1f}], Y[{start_y:.1f}, {start_y + box_height:.1f}]")
    
    # Execute coverage path
    air_node.tool_airgun(True)
    time.sleep(1)
    for i, waypoint in enumerate(path):
        print(f"Waypoint {i + 1}/{len(path)}: x={waypoint[0]:.1f}, y={waypoint[1]:.1f}")
        movel(waypoint, vel=vel, acc=acc)
    air_node.tool_airgun(False)
    time.sleep(1)

    # Return to center (home)
    print("Returning to center...")
    center_pos = posx(center[0], center[1], center[2], rx, ry, rz)
    movel(center_pos, vel=vel, acc=acc)

    print("Coverage path complete!")
    rclpy.shutdown()

if __name__ == "__main__":
    main()
