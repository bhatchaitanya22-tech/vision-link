import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.get_logger().info('Vision-Link Detection Node started')

        self.bridge = CvBridge()
        self.model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')
        self.H = np.load('/home/chaitanya-n-bhat/perception_map/data/calibration/homography_H.npy')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        h, w = frame.shape[:2]
        scale = 800 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = self.model(frame, conf=0.5, verbose=False)
        boxes = results[0].boxes

        self.get_logger().info(f'Detected {len(boxes)} cylinder(s)')

        for i, box in enumerate(boxes):
            cx, cy = map(int, box.xywh[0][:2].tolist())
            conf = float(box.conf[0])

            pixel = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
            mm = cv2.perspectiveTransform(pixel, self.H)[0][0]

            self.get_logger().info(
                f'Cylinder {i+1}: pixel=({cx},{cy}) '
                f'X={mm[0]:.1f}mm Y={mm[1]:.1f}mm '
                f'conf={conf:.2f}'
            )

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()