"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""

from flask import Flask, request, jsonify
import numpy as np
import rclpy
from rclpy.node import Node
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from hday_controller_msgs.srv import ControllerGain
from hday_motion_planner_msgs.srv import Move
import threading
from robot_servers.franka_gripper_server import FrankaGripperServer

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", "127.0.0.1", "URL for the flask server to run on.")
flags.DEFINE_string("ros_port", "11311", "Port for the ROS master to run on.")


class FrankaServer:
    """Handles the starting and stopping of the impedance controller
    (as well as backup) joint recovery policy."""

    def __init__(self, node, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        self.node = node
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target
        self.gripper_type = gripper_type

        self.eepub = self.node.create_publisher(
            PoseStamped, "/hday/fr3_controller/desired_cartesian_pose", 10
        )

        self.gain_client = self.node.create_client(
            ControllerGain,
            "/hday/fr3_controller/gain_serl_cartesian_impedance_controller",
        )
        while not self.gain_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("Waiting for Gain Change Server")

        self.reset_cartesian_impedance_client = self.node.create_client(
            Trigger,
            "/hday/fr3_controller/reset_serl_cartesian_impedance_controller",
        )
        while not self.reset_cartesian_impedance_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.node.get_logger().warn("Waiting for Cartesian Impedance Server")

        self.reset_joint_impedance_client = self.node.create_client(
            Trigger,
            "/hday/fr3_controller/reset_joint_impedance_controller",
        )
        while not self.reset_joint_impedance_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("Waiting for Joint Impedance Server")

        self.motion_planner_client = self.create_client(
            Move, "/hday/engine/motion_planner/move"
        )
        while not self.motion_planner_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Waiting for Motion Planner Server")

        self.eesub = self.node.create_subscription(
            PoseStamped, "/hday/rt_franka/ef_pose", self._set_currpos, 10
        )
        self.jointsub = self.node.create_subscription(
            JointState, "/hday/rt_franka/joint_state", self._set_currjoint, 10
        )
        self.wrenchsub = self.node.create_subscription(
            WrenchStamped, "/hday/sensor/ft_sensor", self._set_currwrench, 10
        )

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        request = Trigger.Request()
        future = self.reset_cartesian_impedance_client.call_async(request)
        print("STOPPING CARTESIAN IMPEDANCE CONTROLLER")
        time.sleep(3)

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.node.get_clock().now().to_msg()
        joint_state_msg.header.frame_id = "link0"
        joint_state_msg.position = self.reset_joint_target
        for i in range(len(joint_state_msg.position)):
            joint_state_msg.name.append("joint" + str(i + 1))

        move_srv = Move.Request()
        move_srv.stamp = joint_state_msg.header.stamp
        move_srv.joint_target = joint_state_msg

        future = self.motion_planner_client.call_async(move_srv)
        print("RUNNING JOINT RESET")

        # Wait until target joint angles are reached
        count = 0
        while True:
            if future.done():
                break
            else:
                print("WAITING MP RESULT TO RESET")
                time.sleep(1)
                count += 1
                if count > 30:
                    print("joint reset TIMEOUT")
                    break

        print("RESET DONE")
        request = Trigger.Request()
        future = self.reset_joint_impedance_client.call_async(request)
        print("STOPPING Joint IMPEDANCE CONTROLLER")
        time.sleep(1)

    def pub_gain(self, param):
        srv = ControllerGain.Request()

        srv.stamp = self.node.get_clock().now().to_msg()
        srv.translational_stiffness = param["translational_stiffness"]
        srv.translational_damping = param["translational_damping"]
        srv.rotational_stiffness = param["rotational_stiffness"]
        srv.rotational_damping = param["rotational_damping"]
        ##### TODO #####
        srv.nullspace_stiffness = 20.0
        srv.nullspace_damping = 9.0
        srv.translational_clip_x = param["translational_clip_x"]
        srv.translational_clip_y = param["translational_clip_y"]
        srv.translational_clip_z = param["translational_clip_z"]
        srv.rotational_clip_x = param["rotational_clip_x"]
        srv.rotational_clip_y = param["rotational_clip_y"]
        srv.rotational_clip_z = param["rotational_clip_z"]

        self.future = self.gain_client.call_async(srv)

    def move(self, pose: list):
        """Moves to a pose: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]

        msg.pose.orientation.x = pose[3]
        msg.pose.orientation.y = pose[4]
        msg.pose.orientation.z = pose[5]
        msg.pose.orientation.w = pose[6]
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        self.pos = np.array(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        #### TODO ####
        self.vel = np.zeros(6)

        jacobian = np.zeros((6, 7))
        self.jacobian = jacobian

    def _set_currjoint(self, msg):
        self.q = np.asarray(msg.position, dtype=np.float32)
        self.dq = np.asarray(msg.velocity, dtype=np.float32)

    def _set_currwrench(self, msg):
        self.force = np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
        )
        self.torque = np.array(
            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]
        )


###############################################################################


def start_ros2_node(node):
    rclpy.spin(node)


def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"

    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    rclpy.init()
    node = Node("franka_server")  # 공통 ROS2 노드

    gripper_server = FrankaGripperServer(node=node)

    """Starts impedance controller"""
    robot_server = FrankaServer(
        node=node,
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )

    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = robot_server.pos[:3]
        r = R.from_quat(robot_server.pos[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        gripper_server.activate_gripper()
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        gripper_server.reset_gripper()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        gripper_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        gripper_server.close()
        return "Closed"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        print("close")
        gripper_server.close_slow()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        gripper_server.move(pos)
        return "Moved Gripper"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        # print("Moving to", pos)
        robot_server.move(pos)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(
            {
                "pose": np.array(robot_server.pos).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper_pos": gripper_server.gripper_pos,
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        ######## TODO #############
        robot_server.pub_gain(request.json)
        return "Updated compliance parameters"

    # ROS2 node 스레드 실행
    ros_thread = threading.Thread(target=start_ros2_node, args=(node,), daemon=True)
    ros_thread.start()

    webapp.run(host=FLAGS.flask_url)


if __name__ == "__main__":
    app.run(main)
