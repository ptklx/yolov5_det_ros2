import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TALKER(Node):
    def __init__(self,name, topic):
        # 初始化
        super().__init__(name)
        #创建发布节点
        self.publisher=self.create_publisher(String, topic, 10)
        self.get_logger().info("TALKER: I'm publisher %s, I have topic '%s'"%(name, topic))
        # 创建定时器
        self.timer = self.create_timer(1.0, self.timer_callback)
        # 定时发布
    def timer_callback(self):
        msg = String()
        msg.data='The talker is publishing ...'
        self.publisher.publish(msg)#发布消息msg
    
def main():
    # 初始化
    rclpy.init()
    # 创建发布节点
    node = TALKER(name='talker_node',topic='yolov_det_ros2')
    # 死循环
    rclpy.spin(node)
    # 关闭
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()