#!/usr/bin/env python3
"""Eye-to-hand calibration of an RGB-D camera (Orbbec Femto) w.r.t. a Kinova arm base.

Collects (base->end-effector, camera->AprilTag) pose pairs via ROS 2 TF, builds the
AX = XB problem and solves it with nonlinear least squares (scipy).

Usage:
  ./calib_femto_kinova.py acquire   [--data-file data.pkl] [--samples-file samples.pkl]
  ./calib_femto_kinova.py calibrate [--data-file data.pkl] [--samples-file samples.pkl] [--visualize]
"""
import rclpy
from rclpy.node import Node

import numpy as np

from scipy.spatial.transform import Rotation as R


import tf2_ros
from tf2_msgs.msg import TFMessage

from geometry_msgs.msg import TransformStamped

from sensor_msgs.msg import Image

import cv2, cv_bridge

import sys
import argparse
import pickle
import termios
import tty

import open3d as o3d


# ---------------------------------------------------------------------------
# SE(3) helpers and AX = XB solver
# ---------------------------------------------------------------------------
def v2t(x):
    """
    Converts a 6D vector (x, y, z, roll, pitch, yaw) into a 4x4 transformation matrix.
    """
    X = np.eye(4)
    rotation = R.from_euler("xyz", x[3:]).as_matrix()

    X[:3, :3] = rotation
    X[:3, 3] = x[:3]

    return X


def t2v(X):
    """
    Converts a 4x4 transformation matrix into a 6D vector (x, y, z, roll, pitch, yaw).
    """
    translation = X[:3, 3]
    rotation = R.from_matrix(X[:3, :3]).as_euler("xyz")

    return np.hstack((translation, rotation))


def solve_eye2hand_NLLS(x_init, data):
    """
    x_init: 4x4 isometry initial guess for camera in robot
    data: tuples (A_i, B_i)_i
    """
    from scipy.optimize import least_squares

    def error_fcn(Ai, Bi, X):
        lhs = Ai @ X
        rhs = X @ Bi

        delta = np.linalg.inv(lhs) @ rhs

        error = t2v(delta)

        return error

    def residuals_fcn(x):
        res = np.zeros(6 * len(data))

        X = v2t(x)

        for i in range(len(data)):
            Ai = data[i][0]
            Bi = data[i][1]

            res[6 * i : 6 * (i + 1)] = error_fcn(Ai, Bi, X)

        return res

    out = least_squares(residuals_fcn, t2v(x_init))

    print(out)
    if out.success:
        return v2t(out.x)
    else:
        return None


class Calibrator(Node):

    def __init__(
        self,
        namespaced_tfs=True,
        init_guess=None,
        tag_frame="tag25h9:0",
        kinova_base_frame="arm_0_base_link",
        kinova_ee_frame="arm_0_end_effector_link",
        femto_frame="rs455_left_base",
    ):
        super().__init__("calib_ouster_femto", namespace="j100_0819")

        self.init_guess = init_guess
        self.calibrated_tf = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tf_topic = "/tf"
        self.tf_static_topic = "/tf_static"

        self.image_topic = "/j100_0819/color/image_rect"
        self.cv_bridge = cv_bridge.CvBridge()

        self.namespace = "j100_0819"

        self.tag_frame = tag_frame
        self.kinova_base_frame = kinova_base_frame
        self.kinova_ee_frame = kinova_ee_frame
        self.femto_frame = femto_frame

        self.data = {"camera_to_tag": [], "base_to_ee": []}
        self.samples = []
        self.images = []

        if namespaced_tfs:

            self.tf_topic = f"/{self.namespace}{self.tf_topic}"
            self.tf_static_topic = f"/{self.namespace}{self.tf_static_topic}"


            old_qos = self.tf_listener.tf_sub.qos_profile
            self.destroy_subscription(self.tf_listener.tf_sub)
            self.tf_listener.tf_sub = self.create_subscription(
                TFMessage,
                self.tf_topic,
                self.tf_listener.callback,
                old_qos,
                callback_group=self.tf_listener.group,
            )
            old_qos_static = self.tf_listener.tf_static_sub.qos_profile
            self.destroy_subscription(self.tf_listener.tf_static_sub)
            self.tf_listener.tf_static_sub = self.create_subscription(
                TFMessage,
                self.tf_static_topic,
                self.tf_listener.static_callback,
                old_qos_static,
                callback_group=self.tf_listener.group,
            )

    def evaluate_consistency(self, guess):

        camera_to_tag = self.data["camera_to_tag"]
        base_to_ee = self.data["base_to_ee"]

        for i in range(len(camera_to_tag)):
            base = np.eye(4, 4)
            ee = base @ base_to_ee[i]
            tag = base @ guess @ camera_to_tag[i]

            tag_to_ee = np.linalg.inv(tag) @ ee
            if i == 0:
                first_tag_to_ee = tag_to_ee
                last_tag_to_ee = tag_to_ee
            if i > 0:
                delta = t2v(tag_to_ee @ np.linalg.inv(first_tag_to_ee))
                delta_t = np.linalg.norm(delta[0:3])
                delta_r = np.linalg.norm(delta[3:6]) * (
                    180.0 / np.pi
                )  # Convert to degrees

                delta_last = t2v(tag_to_ee @ np.linalg.inv(last_tag_to_ee))
                delta_last_t = np.linalg.norm(delta_last[0:3])
                delta_last_r = np.linalg.norm(delta_last[3:6]) * (
                    180.0 / np.pi
                )  # Convert to degrees
                print(
                    f"Data {i}: delta t: {delta_t:.6f} m, delta r: {delta_r:.6f} deg | "
                    f"(w.r.t. last) delta t: {delta_last_t:.6f} m, delta r: {delta_last_r:.6f} deg"
                )
                last_tag_to_ee = tag_to_ee

    def get_tf(
        self,
        source_frame,
        target_frame,
        timeout=10.0,
        sleep_time=0.05,
        max_age=1.5,  # maximum allowed transform age in seconds
    ):
        """
        Returns the current pose of the target_frame with respect to source_frame.
        Only returns transforms that are at most `max_age` seconds old.
        """
        end_time = self.get_clock().now().nanoseconds * 1e-9 + timeout
        last_stamp = None

        while self.get_clock().now().nanoseconds * 1e-9 < end_time:
            rclpy.spin_once(self, timeout_sec=sleep_time)

            try:
                # swap arguments: lookup_transform(target, source, time)
                t: TransformStamped = self.tf_buffer.lookup_transform(
                    source_frame,
                    target_frame,
                    rclpy.time.Time(),
                )

                # check the age of the transform
                now = self.get_clock().now().nanoseconds * 1e-9
                transform_time = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
                age = now - transform_time

                if age <= max_age and (
                    last_stamp is None or t.header.stamp != last_stamp
                ):
                    last_stamp = t.header.stamp
                    print(f"Transform age: {age:.4f}s")
                    return xyz_qxyzw_to_isometry(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                            t.transform.rotation.w,
                        ]
                    )
                else:
                    print("found transform but too old with age:", age)

            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ):
                pass

        self.get_logger().error(
            f"Timeout waiting for fresh transform {source_frame} -> {target_frame}"
        )
        return None

    def snap(self):
        # TODO take care of flipping ambiguity for the Apriltag: https://github.com/AprilRobotics/apriltag/issues/71
        # get direct kinematic
        base_to_ee = self.get_tf(self.kinova_base_frame, self.kinova_ee_frame)
        # get tag in camera
        camera_to_tag = self.get_tf(self.femto_frame, self.tag_frame)

        self._latest_image = None

        def _image_callback(msg):
            self._latest_image = msg

        sub = self.create_subscription(
            Image,
            self.image_topic,
            _image_callback,
            10  # queue size
        )

        end_time = self.get_clock().now().nanoseconds * 1e-9 + 5.0  # timeout 5s
        while self._latest_image is None and self.get_clock().now().nanoseconds * 1e-9 < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

        # cleanup subscription
        self.destroy_subscription(sub)

        msg = self._latest_image

        if base_to_ee is None or camera_to_tag is None:
            self.get_logger().warn(
                f"Could not get transforms {self.kinova_base_frame,}->{self.kinova_ee_frame} or {self.femto_frame}->{self.tag_frame}"
            )
        if msg is not None:
            img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.images.append(img)
        print("base_to_ee\n", base_to_ee)
        print("camera_to_tag\n", camera_to_tag)

        self.data["base_to_ee"].append(base_to_ee)
        self.data["camera_to_tag"].append(camera_to_tag)

    def getch(self):
        """Read a single character from stdin without Enter."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def collect_data(self, data_file="data.pkl", samples_file="samples.pkl"):
        self.get_logger().info("Press ENTER to capture data. Press ESC to stop.")

        while True:
            key = self.getch()

            if key == "\n":  # ENTER
                self.get_logger().info("Capturing tf data. Do not move the robot.")
                self.snap()
                self.get_logger().info("Done")
            elif key == "\x1b":  # ESC
                self.get_logger().info("Stopping")
                break
        assert len(self.data["camera_to_tag"]) == len(self.data["base_to_ee"])
        self.get_logger().info(
            f"Collected {len(self.data['base_to_ee'])} data. Processing data to build AX = XB problem"
        )
        with open(data_file, "wb") as f:
            pickle.dump({"data": self.data}, f)

        n_data = len(self.data["base_to_ee"])
        for i in range(n_data - 1):
            for j in range(i + 1, n_data):
                if self.data["base_to_ee"][i] is None or self.data["camera_to_tag"][i] is None or \
                self.data["base_to_ee"][j] is None or self.data["camera_to_tag"][j] is None:
                    self.get_logger().warn(
                        f"Skipping data {i} because of missing transforms"
                    )
                    continue
                Ai = self.data["base_to_ee"][j] @ np.linalg.inv(
                    self.data["base_to_ee"][i]
                )
                Bi = self.data["camera_to_tag"][j] @ np.linalg.inv(
                    self.data["camera_to_tag"][i]
                )

                self.samples.append((Ai, Bi))

        with open(samples_file, "wb") as f:
            pickle.dump({"samples": self.samples}, f)

        # save all images
        for i, img in enumerate(self.images):
            cv2.imwrite(f"image_{i:03d}.png", img)

    def calibrate(self):

        # try to get it from the tf
        if self.init_guess is None:
            self.init_guess = self.get_tf(self.kinova_base_frame, self.femto_frame)
            if self.init_guess is None:
                self.init_guess = np.eye(4, 4)

        calibration_result = solve_eye2hand_NLLS(
            self.init_guess, self.samples
        )

        print("Calibration result:")
        print(calibration_result)

        self.calibrated_tf = calibration_result

        return calibration_result

    def visualize_data(self, b2e, c2t):
        base = np.eye(4, 4)
        camera = base @ self.init_guess

        camera_calibrated = (
            base @ self.calibrated_tf if self.calibrated_tf is not None else None
        )

        tag = camera @ c2t
        ee = base @ b2e

        geometries = []
        geometries.extend(o3d_isometry(base, "Base (robot)", label_color=[1, 0, 0], size=0.1))
        geometries.extend(o3d_isometry(camera, "Camera (init guess)", label_color=[0, 1, 0], size=0.2))
        geometries.extend(o3d_isometry(tag, "Tag (init guess)", label_color=[0, 0, 1], size=0.1))
        geometries.extend(o3d_isometry(ee, "End Effector", label_color=[1, 1, 0], size=0.1))

        if camera_calibrated is not None:
            tag_calibrated = camera_calibrated @ c2t
            geometries.extend(
                o3d_isometry(camera_calibrated, "Camera (calibrated)", label_color=[0, 1, 1], size=0.1)
            )
            geometries.extend(
                o3d_isometry(tag_calibrated, "Tag (calibrated)", label_color=[0, 0, 0.5], size=0.1)
            )

        # Print legend to console
        print("\n===== Visualization Legend =====")
        print("  \033[91m● Red sphere\033[0m    — Base (robot origin)")
        print("  \033[92m● Green sphere\033[0m  — Camera (initial guess)")
        print("  \033[94m● Blue sphere\033[0m   — Tag (from init guess)")
        print("  \033[93m● Yellow sphere\033[0m — End Effector")
        if camera_calibrated is not None:
            print("  \033[96m● Cyan sphere\033[0m   — Camera (calibrated)")
            print("  \033[34m● DkBlue sphere\033[0m — Tag (calibrated)")
        print("================================\n")

        # Use o3d.visualization.draw() for named geometry sidebar
        try:
            o3d.visualization.draw(geometries, title="Hand-Eye Calibration")
        except AttributeError:
            # Fallback for older Open3D versions without draw()
            flat = [g["geometry"] for g in geometries]
            o3d.visualization.draw_geometries(flat, window_name="Hand-Eye Calibration")


def isometry_to_xyz_qxyzw(T):
    """
    Convert a 4x4 isometry matrix to [x, y, z, qx, qy, qz, qw]
    """
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    q = r.as_quat()  # returns [x, y, z, w]
    return np.hstack((t, q))


def isometry_to_xyz_rpy(T):
    """
    Convert a 4x4 isometry matrix to [x, y, z, r, p, y]
    """
    t = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    a = r.as_euler("xyz", degrees=False)
    return np.hstack((t, a))


def xyz_qxyzw_to_isometry(xyzq):
    """
    Convert [x, y, z, qx, qy, qz, qw] to a 4x4 isometry matrix
    """
    t = xyzq[:3]
    q = xyzq[3:]
    r = R.from_quat(q)
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T


def o3d_isometry(T, name="", label_color=[1, 0, 0], size=0.1):
    # Create coordinate frame
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh_frame.transform(T)

    label_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size * 0.2)
    label_sphere.paint_uniform_color(label_color)
    label_sphere.translate(T[:3, 3] + np.array([0, 0, size * 0.5]))

    return [
        {"name": f"{name} (frame)", "geometry": mesh_frame},
        {"name": f"{name} (label)", "geometry": label_sphere},
    ]


def main():
    parser = argparse.ArgumentParser(description="Femto-Kinova hand-eye calibration")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    acquire_parser = subparsers.add_parser("acquire", help="Interactively collect calibration data")
    acquire_parser.add_argument("--data-file", default="data.pkl", help="Output path for raw data (default: data.pkl)")
    acquire_parser.add_argument("--samples-file", default="samples.pkl", help="Output path for samples (default: samples.pkl)")

    calibrate_parser = subparsers.add_parser("calibrate", help="Run calibration from saved data")
    calibrate_parser.add_argument("--data-file", default="data.pkl", help="Input path for raw data (default: data.pkl)")
    calibrate_parser.add_argument("--samples-file", default="samples.pkl", help="Input path for samples (default: samples.pkl)")
    calibrate_parser.add_argument("--visualize", action="store_true", help="Open 3D visualization after calibration")

    args = parser.parse_args()

    rclpy.init()
    calibrator = Calibrator()

    if args.mode == "acquire":
        calibrator.collect_data(data_file=args.data_file, samples_file=args.samples_file)

    elif args.mode == "calibrate":
        data = pickle.load(open(args.data_file, "rb"))["data"]
        samples = pickle.load(open(args.samples_file, "rb"))["samples"]
        calibrator.data = data
        calibrator.samples = samples

        T = calibrator.calibrate()

        print("\n4x4 Homogeneous matrix:")
        print(T)

        T_rpy = isometry_to_xyz_rpy(T)
        print("\nCamera wrt arm frame:")
        print(f"\nTranslation: x={T_rpy[0]:.6f}  y={T_rpy[1]:.6f}  z={T_rpy[2]:.6f}")
        print(f"Rotation (rad): roll={T_rpy[3]:.6f}  pitch={T_rpy[4]:.6f}  yaw={T_rpy[5]:.6f}")
        print(
            f'\nURDF: <origin xyz="{T_rpy[0]} {T_rpy[1]} {T_rpy[2]}" rpy="{T_rpy[3]} {T_rpy[4]} {T_rpy[5]}"/>'
        )

        T = np.linalg.inv(T)
        T_rpy = isometry_to_xyz_rpy(T)
        print("\nArm wrt camera frame:")
        print(f"\nTranslation: x={T_rpy[0]:.6f}  y={T_rpy[1]:.6f}  z={T_rpy[2]:.6f}")
        print(f"Rotation (rad): roll={T_rpy[3]:.6f}  pitch={T_rpy[4]:.6f}  yaw={T_rpy[5]:.6f}")
        print(
            f'\nURDF: <origin xyz="{T_rpy[0]} {T_rpy[1]} {T_rpy[2]}" rpy="{T_rpy[3]} {T_rpy[4]} {T_rpy[5]}"/>'
        )

        calibrator.evaluate_consistency(T)

        if args.visualize:
            calibrator.visualize_data(
                calibrator.data["base_to_ee"][0],
                calibrator.data["camera_to_tag"][0],
            )

    calibrator.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
