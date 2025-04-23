import rclpy
from rclpy.action import ActionClient
from franka_msgs.action import Grasp, Move
from sensor_msgs.msg import JointState
import numpy as np

from robot_servers.gripper_server import GripperServer


class FrankaGripperServer(GripperServer):
    def __init__(self, node):
        super().__init__()
        self.node = node

        self.gripper_move_client = ActionClient(self.node, Move, "/fr3_gripper/move")
        self.gripper_grasp_client = ActionClient(self.node, Grasp, "/fr3_gripper/grasp")

        self.gripper_sub = self.node.create_subscription(
            JointState, "/franka_gripper/joint_states", self._update_gripper, 10
        )

        self.binary_gripper_pose = 0

    def open(self):
        if self.binary_gripper_pose == 0:
            return

        goal_msg = Move.Goal()
        goal_msg.width = 0.09
        goal_msg.speed = 0.3

        self.gripper_move_client.wait_for_server()
        self.node.get_logger().info("Sending move goal request...")
        self._send_goal_future = self.gripper_move_client.send_goal_async(goal_msg)

        self.binary_gripper_pose = 0

    def close(self):
        if self.binary_gripper_pose == 1:
            return

        goal_msg = Grasp.Goal()
        goal_msg.width = 0.01
        goal_msg.speed = 0.3
        goal_msg.epsilon.inner = 1.0
        goal_msg.epsilon.outer = 1.0
        goal_msg.force = 130.0

        print("A")
        self.gripper_grasp_client.wait_for_server()
        self.node.get_logger().info("Sending grasp goal request...")
        self._send_goal_future = self.gripper_grasp_client.send_goal_async(goal_msg)
        print("B")

        self.binary_gripper_pose = 1

    def close_slow(self):
        if self.binary_gripper_pose == 1:
            return

        goal_msg = Grasp.Goal()
        goal_msg.width = 0.01
        goal_msg.speed = 0.1
        goal_msg.epsilon.inner = 1.0
        goal_msg.epsilon.outer = 1.0
        goal_msg.force = 130.0

        self.gripper_grasp_client.wait_for_server()
        self.node.get_logger().info("Sending grasp goal request...")
        self._send_goal_future = self.gripper_grasp_client.send_goal_async(goal_msg)

        self.binary_gripper_pose = 1

    def move(self, position: int):
        """Move the gripper to a specific position in range [0, 255]"""

        goal_msg = Move.Goal()
        goal_msg.width = float(position / (255 * 10))
        goal_msg.speed = 0.3

        self.gripper_move_client.wait_for_server()
        self.node.get_logger().info("Sending move goal request...")
        self._send_goal_future = self.gripper_move_client.send_goal_async(goal_msg)

    def _update_gripper(self, msg):
        """internal callback to get the latest gripper position."""
        self.gripper_pos = np.sum(msg.position) / 0.08
