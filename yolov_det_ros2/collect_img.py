#!/root/miniforge3/envs/myenv/bin/python3.10
import ctypes
import time
import numpy as np
from multiprocessing import Process, sharedctypes, Lock
import os
import rospy
from std_msgs.msg import Int32,String
from sensor_msgs.msg import Image
import ros_numpy
from cv_bridge import CvBridge
import cv2
from yolov5_det import yolov5det ,draw_detect_res,customized_filtration,left2right_point ,pcl_lenghs

color_frame_mem = sharedctypes.RawArray(ctypes.c_uint8, 921600)
depth_frame_mem = sharedctypes.RawArray(ctypes.c_int, 307200)
status_mem = sharedctypes.RawArray(ctypes.c_uint8, 2)  ############
result_mem = sharedctypes.RawArray(ctypes.c_int, 5)

color_frame = np.frombuffer(color_frame_mem, dtype=ctypes.c_uint8).reshape(480,640,3)
depth_frame = np.frombuffer(depth_frame_mem, dtype=ctypes.c_int).reshape(480,640)
status = np.frombuffer(status_mem, dtype=ctypes.c_uint8)
color_frame_lock = Lock()
depth_frame_lock = Lock()
result = np.frombuffer(result_mem, dtype=ctypes.c_int)



def compute_command_callback(msg):  # 1 
    # rospy.loginfo(rospy.get_caller_id() + "I heard command: %s", msg.data)
    print(f"i heard :{msg.data}")
    result[0]= msg.data

# def class_command_callback(msg):
#     # rospy.loginfo(rospy.get_caller_id() + "I heard command: %s", msg.data)
#     print(f"i heard class :{msg.data}")
#     result[1]= msg.data

def color_image_callback(msg):
    bridge = CvBridge()
    color_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    color_frame_lock.acquire()
    color_frame[:] = color_image
    status[0] = 1
    color_frame_lock.release()


def depth_image_callback(msg):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    depth_frame_lock.acquire()
    depth_frame[:] = depth_image
    status[1] = 1
    depth_frame_lock.release()


# def start_recording(self):
#     self.update_result_text("准备录音...")  # 在开始录音之前更新状态
#     subprocess.call(['python3', '/home/aidlux/ros_ws/src/my_arm_control_pkg/scripts/test.py'])  # 调用test.py脚本来执行录音

# def start_recording_route(self):
#     threading.Thread(target=self.start_recording).start()  # 在后台线程中启动录音
#     return "Recording started", 200




def center_manager():
    rospy.init_node('center_manager', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, color_image_callback)
    rospy.Subscriber('/camera/depth/image_rect_raw', Image, depth_image_callback)
    # rospy.Subscriber('/camera/rgb/image_color', Image, color_image_callback)
    # rospy.Subscriber('/camera/depth/image', Image, depth_image_callback)
    
    rospy.Subscriber('/robot/arm/commond', Int32, compute_command_callback)
    # rospy.Subscriber('/robot/voice/commond', Int32, class_command_callback)

    # pub_starter = rospy.Publisher('/ai/detecter/start', String, queue_size=1)
    rate = rospy.Rate(15)
    folder_name = './save_images'
    index=0 
    while not rospy.is_shutdown():
        if result[0] != -1:
            # result[0] = -1  #  test once 
        #     if st >= time_st:
        #         result[4] = 0
        #         st = 0
        #     elif result[4] > 0 and st < time_st:
        #         st += 1
            print("start detect !")
            ## get image
            color_frame_lock.acquire()
            frame = color_frame.copy()
            status[0] = 0
            color_frame_lock.release()
            depth_frame_lock.acquire()
            dframe = depth_frame.copy()
            status[1] = 0
            depth_frame_lock.release()
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            image_path = os.path.join(folder_name, f'new_{index}.png')
            cv2.imwrite(image_path, frame)
            index+=1
            result[0] = -1
            # pub_starter.publish(pub_start)
            rate.sleep()
            
        
if __name__ == "__main__":
    try:
        center_manager()
    except rospy.ROSInterruptException:
        pass