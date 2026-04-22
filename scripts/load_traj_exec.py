import rclpy
import DR_init
import sys
import time
from tool_control import ToolControlNode
import pandas as pd
import numpy as np

CSV_PATH = "~/doosan_pose_list.csv"


rclpy.init()
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL
node = rclpy.create_node('coverage_path', namespace=ROBOT_ID)
DR_init.__dsr__node = node

from DSR_ROBOT2 import movej, posj, movel, posx, set_robot_mode, get_current_posx, ROBOT_MODE_AUTONOMOUS

set_robot_mode(ROBOT_MODE_AUTONOMOUS)

def move_to_init():
    print("moving to initial point")
    movej(posj(90, 0, -90, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-70, 0, -90, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-70, -30, -60, 0, 0, 0), vel=60, acc=60)
    time.sleep(1)
    movej(posj(-75, -30, -60, 0, -90, 0), vel=60, acc=60)
    time.sleep(1)
    print("moved to initial point")

def read_from_csv_and_exec(csv_path):
    vel = 30
    acc = 30

    print("start reading csv")
    pose_df = pd.read_csv(csv_path)
    pose_np = np.array(pose_df)

    print("execute the pose")
    for i, pose in enumerate(pose_np):
        print("executing ", str(i), " th pose: ", pose)
        movel(posx(pose.tolist()), vel=vel, acc=acc)
        print("executed ", str(i))
        # time.sleep(0.5)


def main(args=None):
    move_to_init()
    read_from_csv_and_exec(CSV_PATH)

    rclpy.shutdown()

if __name__ == "__main__":
    main()

