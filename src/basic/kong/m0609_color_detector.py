#!/usr/bin/env python3
"""
PC B에서 실행하는 M0609 Wrist Camera 색상 감지 노드.

입력:
    /wrist_camera/rgb          sensor_msgs/msg/Image

출력:
    /detected_cube_color       std_msgs/msg/Int32
        0 = 미검출
        1 = 파란색
        2 = 초록색

선택 출력:
    /wrist_camera/color_debug  sensor_msgs/msg/Image
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


COLOR_NONE = 0
COLOR_BLUE = 1
COLOR_GREEN = 2

COLOR_NAMES = {
    COLOR_NONE: "NONE",
    COLOR_BLUE: "BLUE",
    COLOR_GREEN: "GREEN",
}


class M0609ColorDetector(Node):
    def __init__(self) -> None:
        super().__init__("m0609_color_detector")

        # ----------------------------------------------------
        # ROS 파라미터
        # ----------------------------------------------------
        self.declare_parameter("image_topic", "/wrist_camera/rgb")
        self.declare_parameter("result_topic", "/detected_cube_color")
        self.declare_parameter("debug_topic", "/wrist_camera/color_debug")

        self.declare_parameter("min_area", 450.0)
        self.declare_parameter("roi_ratio", 0.72)
        self.declare_parameter("dominance_ratio", 1.15)
        self.declare_parameter("publish_debug", True)

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )
        self.debug_topic = (
            self.get_parameter("debug_topic").get_parameter_value().string_value
        )

        self.min_area = (
            self.get_parameter("min_area").get_parameter_value().double_value
        )
        self.roi_ratio = (
            self.get_parameter("roi_ratio").get_parameter_value().double_value
        )
        self.dominance_ratio = (
            self.get_parameter("dominance_ratio").get_parameter_value().double_value
        )
        self.publish_debug = (
            self.get_parameter("publish_debug").get_parameter_value().bool_value
        )

        if not 0.1 <= self.roi_ratio <= 1.0:
            raise ValueError("roi_ratio는 0.1~1.0 범위여야 합니다.")

        # ----------------------------------------------------
        # HSV 범위
        # OpenCV Hue 범위는 0~179이다.
        # ----------------------------------------------------
        self.hsv_ranges: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            COLOR_BLUE: (
                np.array([90, 70, 45], dtype=np.uint8),
                np.array([140, 255, 255], dtype=np.uint8),
            ),
            COLOR_GREEN: (
                np.array([35, 60, 40], dtype=np.uint8),
                np.array([90, 255, 255], dtype=np.uint8),
            ),
        }

        self.bridge = CvBridge()
        self.kernel = np.ones((5, 5), dtype=np.uint8)

        # Image는 센서 데이터 QoS(Best Effort)를 사용한다.
        self.image_subscriber = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        # 판별 결과는 신뢰성 있는 작은 메시지이므로 Reliable QoS를 사용한다.
        result_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.result_publisher = self.create_publisher(
            Int32,
            self.result_topic,
            result_qos,
        )

        self.debug_publisher: Optional[rclpy.publisher.Publisher] = None
        if self.publish_debug:
            self.debug_publisher = self.create_publisher(
                Image,
                self.debug_topic,
                qos_profile_sensor_data,
            )

        self.last_logged_result: Optional[int] = None

        self.get_logger().info(
            f"구독: {self.image_topic} | "
            f"결과: {self.result_topic} | "
            f"debug: {self.publish_debug}"
        )
        self.get_logger().info(
            "색상 코드: 0=NONE, 1=BLUE, 2=GREEN"
        )

    @staticmethod
    def largest_contour(
        mask: np.ndarray,
    ) -> Tuple[float, Optional[np.ndarray]]:
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return 0.0, None

        contour = max(contours, key=cv2.contourArea)
        return float(cv2.contourArea(contour)), contour

    def create_mask(
        self,
        hsv_roi: np.ndarray,
        color_id: int,
    ) -> np.ndarray:
        lower, upper = self.hsv_ranges[color_id]
        mask = cv2.inRange(hsv_roi, lower, upper)

        # 작은 노이즈 제거 후 분리된 영역을 연결한다.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=1,
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=2,
        )
        return mask

    def decide_color(
        self,
        areas: Dict[int, float],
    ) -> int:
        blue_area = areas[COLOR_BLUE]
        green_area = areas[COLOR_GREEN]

        if blue_area < self.min_area and green_area < self.min_area:
            return COLOR_NONE

        if blue_area >= self.min_area and green_area < self.min_area:
            return COLOR_BLUE

        if green_area >= self.min_area and blue_area < self.min_area:
            return COLOR_GREEN

        # 두 색이 동시에 보일 때는 충분히 우세한 색만 선택한다.
        if blue_area >= green_area * self.dominance_ratio:
            return COLOR_BLUE

        if green_area >= blue_area * self.dominance_ratio:
            return COLOR_GREEN

        return COLOR_NONE

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except CvBridgeError as exc:
            self.get_logger().error(f"cv_bridge 변환 실패: {exc}")
            return

        if frame is None or frame.size == 0:
            self.get_logger().warning("빈 이미지가 수신되었습니다.")
            return

        height, width = frame.shape[:2]

        roi_width = max(1, int(width * self.roi_ratio))
        roi_height = max(1, int(height * self.roi_ratio))

        x0 = (width - roi_width) // 2
        y0 = (height - roi_height) // 2
        x1 = x0 + roi_width
        y1 = y0 + roi_height

        roi = frame[y0:y1, x0:x1]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        areas: Dict[int, float] = {}
        contours: Dict[int, Optional[np.ndarray]] = {}

        for color_id in (COLOR_BLUE, COLOR_GREEN):
            mask = self.create_mask(hsv_roi, color_id)
            area, contour = self.largest_contour(mask)
            areas[color_id] = area
            contours[color_id] = contour

        detected_color = self.decide_color(areas)

        result_msg = Int32()
        result_msg.data = detected_color
        self.result_publisher.publish(result_msg)

        if detected_color != self.last_logged_result:
            self.get_logger().info(
                f"판별={COLOR_NAMES[detected_color]}({detected_color}) | "
                f"blue_area={areas[COLOR_BLUE]:.0f}, "
                f"green_area={areas[COLOR_GREEN]:.0f}"
            )
            self.last_logged_result = detected_color

        if self.debug_publisher is not None:
            debug_frame = frame.copy()

            cv2.rectangle(
                debug_frame,
                (x0, y0),
                (x1, y1),
                (255, 255, 255),
                2,
            )

            contour_colors = {
                COLOR_BLUE: (255, 0, 0),    # BGR
                COLOR_GREEN: (0, 255, 0),
            }

            for color_id in (COLOR_BLUE, COLOR_GREEN):
                contour = contours[color_id]
                if contour is None:
                    continue

                shifted_contour = contour.copy()
                shifted_contour[:, 0, 0] += x0
                shifted_contour[:, 0, 1] += y0

                cv2.drawContours(
                    debug_frame,
                    [shifted_contour],
                    contourIdx=-1,
                    color=contour_colors[color_id],
                    thickness=2,
                )

                bx, by, bw, bh = cv2.boundingRect(shifted_contour)
                cv2.rectangle(
                    debug_frame,
                    (bx, by),
                    (bx + bw, by + bh),
                    contour_colors[color_id],
                    2,
                )

            label = (
                f"{COLOR_NAMES[detected_color]} ({detected_color})  "
                f"B:{areas[COLOR_BLUE]:.0f}  "
                f"G:{areas[COLOR_GREEN]:.0f}"
            )
            cv2.putText(
                debug_frame,
                label,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            try:
                debug_msg = self.bridge.cv2_to_imgmsg(
                    debug_frame,
                    encoding="bgr8",
                )
                debug_msg.header = msg.header
                self.debug_publisher.publish(debug_msg)
            except CvBridgeError as exc:
                self.get_logger().error(f"Debug 이미지 변환 실패: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = M0609ColorDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()