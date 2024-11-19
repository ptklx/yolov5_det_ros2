import ctypes
import numpy as np
import time
from multiprocessing import sharedctypes, Lock
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32,String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError
import cv2
import json

from yolov5_det import yolov5det ,draw_detect_res,middleorleft2right_point,\
    customized_filtration_min_max ,get_mid_pos ,draw_circle_res,get_soda_beer_mid
from yolov5_soda_beer_w import yolov5det as yolov5_soda_beer

#soda=0  beer=1  water = 2 coke=3   fanta=4
os.environ['ADSP_LIBRARY_PATH'] = '/usr/local/lib/aidlux/aidlite/snpe2/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'
class YOLOManager(Node):
    def __init__(self):
        super().__init__('yolo_manager')
        self.get_logger().info("start yolo object detect!!!")
        color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 921600)
        depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 307200)

        # color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 2764800)  #720 1280
        # depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 921600)

        status_mem = sharedctypes.RawArray(ctypes.c_uint8, 2)  ############
        result_mem = sharedctypes.RawArray(ctypes.c_int, 5)

        self.color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(480,640,3)
        self.depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(480,640)

        # self.color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(720,1280,3)
        # self.depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(720,1280)

        self.status = np.frombuffer(status_mem, dtype=ctypes.c_uint8)
        self.color_frame_lock = Lock()
        self.depth_frame_lock = Lock()
        self.result = np.frombuffer(result_mem, dtype=ctypes.c_int)

        self.model =yolov5det(os.path.join(os.path.dirname(os.path.abspath(__file__)),'models/cutoff_8550_det_object_detection_best_fp16.qnn216.ctx.bin'))
        path_onnx = os.path.join(os.path.dirname(os.path.abspath(__file__)),"models/cutoff_8550_soda_beer_water_fp16.qnn216.ctx.bin")
        self.model_so_b = yolov5_soda_beer(path_onnx)

        self.subscription_color = self.create_subscription(
            Image,
            '/cam_1/cam_1/color/image_raw',
            self.color_image_callback,
            10
        )
        self.subscription_depth = self.create_subscription(
            Image,
           '/cam_1/cam_1/aligned_depth_to_color/image_raw', 
            self.depth_image_callback,
            10
        )
        self.subscription_command = self.create_subscription(
            String,
            '/ws/ai/detecter',
            self.compute_command_callback,
            10
        )
        self.publisher_result = self.create_publisher(String, '/ai/detecter/result', 10)
       
        self.time_st = 30  ###
        self.st =0
        self.result[2]=1
        self.result_str = "{},{},{}".format(0,0,0)
        self.timer = self.create_timer(1.0 / 20.0, self.timer_callback)

    def color_image_callback(self, msg):
        try:
            bridge = CvBridge()
            color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv2.imwrite("pt_color.png", color_image)
            self.color_frame_lock.acquire()
            self.color_frame[:] = color_image
            self.status[0] = 1
            self.color_frame_lock.release()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        else:
            # cv2.imwrite("pt_color.png", color_image)
            pass

    def depth_image_callback(self, msg):
        try:
            bridge = CvBridge()
            # 将 ROS 图像消息转换为 OpenCV 图像
            # cv_image = bridge.imgmsg_to_cv2(msg, "16UC1")
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_frame_lock.acquire()
            self.depth_frame[:] = depth_image
            self.status[1] = 1
            self.depth_frame_lock.release()
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridgeError: {e}")
        else:
            # 在这里处理深度图像
            # cv2.imwrite("depth.png", depth_image)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # cv2.imwrite("depth.png", depth_colormap)
            pass


    def compute_command_callback(self, msg):
        # self.get_logger().info(f"Received command: {msg.data}")
        print(f"i heard :{msg.data}")
        try :
            data = json.loads(msg.data)
            need_class = data['message']
            self.result[0]= need_class
        except   CvBridgeError as e:
            print("the detector not json string")


    def timer_callback(self):
        if  self.result[0] != -1 and self.st==0:
            need_class =  self.result[0]-1  if  self.result[0]>0  else  self.result[0]
            print("start detect !")
            ## get image
            self.color_frame_lock.acquire()
            frame =  self.color_frame.copy()
            self.status[0] = 0
            self.color_frame_lock.release()
            self.depth_frame_lock.acquire()
            dframe =  self.depth_frame.copy()
            self.status[1] = 0
            self.depth_frame_lock.release()
            ## det
            t1=time.time()
            preds = self.model(frame)
            cost_time = (time.time()-t1)*1000
            # print(f"first inference time {cost_time}")

            if len(preds):
                det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
                
                showframe = draw_detect_res(frame, preds)   ### 测试
                if need_class==0:
                    det_pred_soda=self.model_so_b(frame)  
                    mid_soda_beer_x = get_soda_beer_mid(det_pred_soda)
                    if self.result[0]==0:
                        one_point = middleorleft2right_point(det_pred,(frame.shape[0],frame.shape[1]),
                                                             need_class=need_class,mid_soda_beer=mid_soda_beer_x,soda_beer='soda')  # x,y
                    else:
                        one_point = middleorleft2right_point(det_pred,(frame.shape[0],frame.shape[1]),
                                                             need_class=need_class,mid_soda_beer=mid_soda_beer_x,soda_beer='beer')  # x,y
                else:
                    one_point = middleorleft2right_point(det_pred,(frame.shape[0],frame.shape[1]),need_class=need_class)
                if one_point==[]:
                    print(" no choose object !")
                    self.result[0] = -1
                    cv2.imwrite("box_result.png", showframe)
                    return
                circle_frame = draw_circle_res(showframe,one_point)
                cv2.imwrite("box_result.png", circle_frame)
                mean_v =get_mid_pos(one_point,dframe,24)
                # tmp_str = "[{},{},{}]".format(int(one_point[0]),int(one_point[1]),mean_v)
                # result_str = result_str.replace("[0,0,0]",tmp_str)
                self.result_str = "{},{},{}".format(int(one_point[0]),int(one_point[1]),mean_v)
                print(self.result_str)
            else:
                self.result_str = "{},{},{}".format(0,0,0)
                print("not detect object")
            self.result[2]=1
            msg = String()
            msg.data = self.result_str
            self.publisher_result.publish(msg)
            self.result[0] = -1
        else:
            if self.result[2] !=0:
                if self.st < self.time_st:
                    self.st += 1
                if self.st >= self.time_st:
                    msg = String()
                    msg.data ="{},{},{}".format(0,0,0)
                    self.publisher_result.publish(msg)
                    self.result[2] = 0
                    self.st = 0



def center_manager():
    rclpy.init()
    node = YOLOManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
if __name__ == "__main__":
    try:
        center_manager()
    except Exception as e:
        print(f"Exception occurred :{e}")