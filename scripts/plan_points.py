# plan_points.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from visualization_msgs.msg import InteractiveMarkerFeedback
from geometry_msgs.msg import Pose, Vector3
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
import rclpy.time
import time

class WaypointRunner(Node):
    def __init__(self):
        super().__init__('waypoint_runner')
        self._client = ActionClient(self, MoveGroup, '/move_action')

        # TF lookup 준비
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # RViz goal state 구독
        self.goal_pose = None
        self.create_subscription(
            InteractiveMarkerFeedback,
            '/rviz_moveit_motion_planning_display/robot_interaction_interactive_marker_topic/feedback',
            self.goal_callback,
            10
        )

    def goal_callback(self, msg):
        if 'goal' in msg.marker_name:
            self.goal_pose = msg.pose

    def get_current_pose(self) -> Pose:
        deadline = self.get_clock().now() + Duration(seconds=5.0)
        while self.get_clock().now() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                t = self.tf_buffer.lookup_transform('world', 'link_6', rclpy.time.Time())
                pose = Pose()
                pose.position.x = t.transform.translation.x
                pose.position.y = t.transform.translation.y
                pose.position.z = t.transform.translation.z
                pose.orientation.x = t.transform.rotation.x
                pose.orientation.y = t.transform.rotation.y
                pose.orientation.z = t.transform.rotation.z
                pose.orientation.w = t.transform.rotation.w
                return pose
            except Exception:
                continue
        raise RuntimeError("TF lookup 실패")

    def wait_for_goal_pose(self) -> Pose:
        print("\nRViz에서 goal state를 드래그로 설정하세요.")
        print("설정 완료 후 Enter를 누르세요...")

        self.goal_pose = None
        import threading
        entered = threading.Event()
        threading.Thread(target=lambda: [input(), entered.set()], daemon=True).start()

        while not entered.is_set():
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.goal_pose is None:
            raise RuntimeError("Goal pose를 받지 못했습니다. RViz에서 마커를 드래그했는지 확인하세요.")

        return self.goal_pose

    def move_to_pose(self, pose: Pose):
        self._client.wait_for_server()

        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = "manipulator"
        req.num_planning_attempts = 10
        req.allowed_planning_time = 10.0        # 5.0 → 10.0
        req.max_velocity_scaling_factor = 0.1   # 0.3 → 0.1
        req.max_acceleration_scaling_factor = 0.1  # 0.3 → 0.1

        # Position constraint
        pos_con = PositionConstraint()
        pos_con.header.frame_id = "world"
        pos_con.link_name = "link_6"
        pos_con.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.SPHERE
        prim.dimensions = [0.01]
        bv = BoundingVolume()
        bv.primitives.append(prim)
        bv.primitive_poses.append(pose)
        pos_con.constraint_region = bv
        pos_con.weight = 1.0

        # Orientation constraint
        ori_con = OrientationConstraint()
        ori_con.header.frame_id = "world"
        ori_con.link_name = "link_6"
        ori_con.orientation = pose.orientation
        ori_con.absolute_x_axis_tolerance = 0.1
        ori_con.absolute_y_axis_tolerance = 0.1
        ori_con.absolute_z_axis_tolerance = 0.1
        ori_con.weight = 1.0

        con = Constraints()
        con.position_constraints.append(pos_con)
        con.orientation_constraints.append(ori_con)
        req.goal_constraints.append(con)

        goal.request = req

        # Goal 전송 (타임아웃 15초)
        future = self._client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Goal이 거절됐습니다!")
            return

        # 결과 대기 (타임아웃 60초)
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=60.0)

        if not result_future.done():
            self.get_logger().warn("결과 수신 타임아웃 — 하지만 실행은 완료됐을 수 있습니다")
            return

        error_code = result_future.result().result.error_code.val
        if error_code == 1:
            self.get_logger().info("성공!")
        else:
            self.get_logger().warn(f"실패 (error_code: {error_code})")

def main():
    rclpy.init()
    node = WaypointRunner()

    # 1. 현재 로봇 위치 읽기
    current = node.get_current_pose()
    node.get_logger().info(
        f"Start pose → "
        f"x={current.position.x:.3f}, y={current.position.y:.3f}, z={current.position.z:.3f}"
    )

    # 2. RViz에서 goal state 드래그 후 Enter
    goal = node.wait_for_goal_pose()
    node.get_logger().info(
        f"Goal pose → "
        f"x={goal.position.x:.3f}, y={goal.position.y:.3f}, z={goal.position.z:.3f}"
    )

    # 3. 경로 계획 및 실행
    node.get_logger().info("경로 계획 및 실행 중...")
    node.move_to_pose(goal)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
