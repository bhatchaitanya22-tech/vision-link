import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
from ultralytics import YOLO

CYLINDER_HEIGHT_MM = 50.0

class GraspPublisher(Node):
    def __init__(self):
        super().__init__('grasp_publisher')
        self.get_logger().info('Vision-Link Grasp Publisher started')

        self.publisher = self.create_publisher(PoseStamped, '/grasp/target', 10)
        self.timer = self.create_timer(2.0, self.publish_grasp)

        self.model = YOLO('/home/chaitanya-n-bhat/perception_map/06_project/best.pt')
        self.H = np.load('/home/chaitanya-n-bhat/perception_map/data/calibration/homography_H.npy')

        self.img_path = '/home/chaitanya-n-bhat/perception_map/data/raw/batch_03/IMG_20260321_155933.jpg'

    def publish_grasp(self):
        img = cv2.imread(self.img_path)
        h, w = img.shape[:2]
        scale = 800 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        results = self.model(img, conf=0.5, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            self.get_logger().warn('No cylinders detected')
            return

        # Pick the highest confidence detection
        best = max(boxes, key=lambda b: float(b.conf[0]))
        cx, cy = map(int, best.xywh[0][:2].tolist())
        conf = float(best.conf[0])

        pixel = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
        mm = cv2.perspectiveTransform(pixel, self.H)[0][0]

        # Build PoseStamped message
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        # Convert mm to metres for ROS2
        msg.pose.position.x = float(mm[0]) / 1000.0
        msg.pose.position.y = float(mm[1]) / 1000.0
        msg.pose.position.z = float(CYLINDER_HEIGHT_MM) / 1000.0

        # Upright cylinder — no rotation
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)
        self.get_logger().info(
            f'Grasp target published: '
            f'X={mm[0]:.1f}mm Y={mm[1]:.1f}mm Z={CYLINDER_HEIGHT_MM}mm '
            f'conf={conf:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = GraspPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()