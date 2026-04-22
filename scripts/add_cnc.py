import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Pose, Point
import trimesh
import numpy as np
import time
import math

class CollisionObjectAdder(Node):
    def __init__(self):
        super().__init__('cnc_collision_adder')
        self.pub = self.create_publisher(
            CollisionObject,
            '/collision_object',
            10
        )

    def add_mesh(self, obj_path: str, pose: Pose, object_id: str = "cnc_machine"):
        # OBJ 로드 (멀티 메쉬 대응)
        loaded = trimesh.load(obj_path)

        # Scene이면 모든 메쉬를 하나로 합치기
        if isinstance(loaded, trimesh.Scene):
            mesh_data = trimesh.util.concatenate(
                [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        else:
            mesh_data = loaded

        # mm → m 변환
        mesh_data.vertices /= 1000.0

        self.get_logger().info(f"Mesh loaded: {len(mesh_data.vertices)} vertices, {len(mesh_data.faces)} faces")
        self.get_logger().info(f"Bounds (m): {mesh_data.bounds}")

        co = CollisionObject()
        co.header.frame_id = "world"
        co.id = object_id
        co.operation = CollisionObject.ADD

        ros_mesh = Mesh()
        for v in mesh_data.vertices:
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            ros_mesh.vertices.append(p)
        for f in mesh_data.faces:
            t = MeshTriangle()
            t.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
            ros_mesh.triangles.append(t)

        co.meshes.append(ros_mesh)
        co.mesh_poses.append(pose)

        self.get_logger().info("Publishing collision object...")
        for _ in range(10):
            self.pub.publish(co)
            time.sleep(0.5)
        self.get_logger().info("Done")


def main():
    rclpy.init()
    node = CollisionObjectAdder()

    pose = Pose()
    pose.position.x = -0.61
    pose.position.y = 0.365
    pose.position.z = -0.91
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 1.0   
    pose.orientation.w = 0.0
    
    obj_path = "/home/dsb/ros2_ws/src/KIMM-doosan-robot2/scripts/meshes/VMC-300-l.obj"
    node.add_mesh(obj_path, pose)

    rclpy.shutdown()

if __name__ == "__main__":
    main()
