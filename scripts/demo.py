import rclpy
import DR_init
import sys
import time

def main(args=None):
    rclpy.init(args=args)

    ROBOT_ID = ""
    ROBOT_MODEL = "m0609"
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    node = rclpy.create_node('example_py', namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    from DSR_ROBOT2 import movej, posj, movel, posx, set_robot_mode, get_current_posx,  ROBOT_MODE_AUTONOMOUS

    set_robot_mode(ROBOT_MODE_AUTONOMOUS)

    target_pos = posj(0, 0, 0, 0, 0, 0)
    movej(target_pos, vel=100, acc=100)

    time.sleep(2)

    cur_x = get_current_posx()[0]

    print(cur_x)
    target_x = posx(cur_x[0], cur_x[1], cur_x[2]-20, cur_x[3], cur_x[4], cur_x[5])
    movel(target_x, vel=30, acc=60)
    print("target1: ", target_x)
    time.sleep(2)
    cur_x = get_current_posx()
    cur_x = cur_x[0]
    print("cur1: ", cur_x)
    target_x = posx(cur_x[0], cur_x[1]+10, cur_x[2], 0,0,0)
    print("target2: ", target_x)
    movel(target_x, vel=30, acc=60)
    print("cur2: ", cur_x)
    

    print("Example complete")
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
