#!/usr/bin/python3
#-*- encoding: utf-8 -*-
import os.path as ops
import numpy as np
import torch
import cv2
import time
import math
import tf
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
# from dataset.dataset_utils import TUSIMPLE
from model2 import *
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from driving_es import *
from slidewindow import *
from geometry_msgs.msg import  Vector3
from lidar_object_detection.msg import ObjectInfo
from visualization_msgs.msg import MarkerArray
from move_base_msgs.msg import MoveBaseActionResult
from std_msgs.msg import Float64, Int64, Bool
from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage
from race.msg import drive_values


import csv


# drive_values_pub= rospy.Publisher('control_value', drive_values, queue_size = 1)

class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


class PID():
  def __init__(self,kp,ki,kd):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.p_error = 0.0
    self.i_error = 0.0
    self.d_error = 0.0

  def pid_control(self, cte):
    self.d_error = cte-self.p_error
    self.p_error = cte
    self.i_error += cte

    return self.kp*self.p_error + self.ki*self.i_error + self.kd*self.d_error
  
class LanenetDetection:
    def __init__(self):
        rospy.init_node('lanenet_detection_node')

        rospy.Subscriber("/usb_cam/image_raw", Image, self.camCB)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)

        self.ctrl_cmd_pub = rospy.Publisher('control_value', drive_values, queue_size=1)

        # print ("----- Xycar self driving -----")

        self.drive_value = drive_values()


        
        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])
        self.slidewindow = SlideWindow()
        self.x_location = 320
        self.last_x_location = 320
        self.is_detected = True
        self.current_lane = "LEFT"
        self.last_angle = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = '/home/foscar/iscc_2024/src/camera/lanenet/src/lanenet_.model' #내동영상으로 바꿔준다
        self.LaneNet_model = Lanenet(2, 4)
        self.LaneNet_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.LaneNet_model.to(self.device)



        # 데이터 저장 경로 및 CSV 파일 설정
        self.data_dir = '/home/foscar/iscc_2024/src/camera/lanenet/data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, 'lane_data.csv')
        self.init_csv()



        self.x_location = 320
        self.last_x_location = 320

        self.motor_msg = 0

        #------------------------------------- 동적 장애물 파라미터 -------------------------------------#
        self.tunnel_dynamic_check_flag = False
        self.is_dynamic = False
        self.obstacle_list = []
        self.theta_list = []
        var_rate = 20

        self.tunnel_dynamic_steering_list = []
        self.tunnel_dynamic_steering_list_max_len = 30
        self.steer_diff = -1
        self.steer_diff_threshold = 5
        self.is_steer_stable = False

        self.tunnel_dynamic_obstacle_y_list = []
        self.tunnel_dynamic_obstacle_y_list_max_len = 50
        self.obstacle_y_diff = -1
        self.obstacle_y_diff_threshold = 0.2

        #------------------------------------- 정적 장애물 파라미터 --------------------------------------#
        tunnel_statlc_roi_check_arr = [0, 0]
        self.tunnel_static_half_flag = False


        rate = rospy.Rate(var_rate)
        while not rospy.is_shutdown():
            try:
                
                pid = PID(0.2, 0.05, 0.01)
                cv2.imshow("lane_image", self.image)

                show_img = self.inference(self.image)
                # init_show_img(show_img)

                cv2.imshow("show", show_img)

                img_frame = show_img.copy() # img_frame변수에 카메라 이미지를 받아옵니다.   
                height,width,channel = img_frame.shape # 이미지의 높이,너비,채널값을 변수에 할당합니다. 
                
                
                img_roi = img_frame[280:470,0:]   # y좌표 0~320 사이에는 차선과 관련없는 이미지들이 존재하기에 노이즈를 줄이기 위하여 roi설정을 해주었습니다.




                img_filtered = color_filter(img_roi)   #roi가 설정된 이미지를 color_filtering 하여 흰색 픽셀만을 추출해냅니다. 

                height, width, channel = img_filtered.shape
                # print(f"width: {width}\nheight: {height}") # 640, 480


                cv2.imshow("filter", img_filtered)
                
                
                # img_warped = bird_eye_view(img_filtered,width,height) # 앞서 구현한 bird-eye-view 함수를 이용하여 시점변환해줍니다. 


                # 예전 bird eye view
                # left_margin = 195
                # top_margin =  44
                # src_point1 = [0, 190]      # 왼쪽 아래
                # src_point2 = [left_margin, top_margin]
                # src_point3 = [width-left_margin, top_margin]
                # src_point4 = [width , 190]  

                # src_points = np.float32([src_point1, src_point2, src_point3, src_point4])
                


                # dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
                
                # matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                # img_warped = cv2.warpPerspective(img_filtered, matrix, [width, height])

                # 고친 bird eye view 
                left_margin = 180
                top_margin =  44
                src_point1 = [0, 190]      # 왼쪽 아래
                src_point2 = [left_margin, top_margin]      #195,44
                src_point3 = [width-left_margin, top_margin]    #445,44
                src_point4 = [width , 190]  #640, 190

                src_points = np.float32([src_point1, src_point2, src_point3, src_point4])
                
                # dst_point1 = [120, 190]    # 왼쪽 아래     #160,190
                # dst_point2 = [220, 10]      # 왼쪽 위       #160,0
                # dst_point3 = [420, 10]    # 오른쪽 위     #480,0
                # dst_point4 = [480, 200]  # 오른쪽 아래   #480,0

                dst_point1 = [width//4, 190]    # 왼쪽 아래
                dst_point2 = [width//4, 0]      # 왼쪽 위 160
                dst_point3 = [width//4*3, 0]    # 오른쪽 위   480
                dst_point4 = [width//4*3, 190]  # 오른쪽 아래

                dst_points = np.float32([dst_point1, dst_point2, dst_point3, dst_point4])
                
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                img_warped = cv2.warpPerspective(img_filtered, matrix, [width, height])

                img_warped = cv2.resize(img_warped,dsize=(640,480))
                cv2.imshow('img_warped', img_warped)

                if self.tunnel_static_half_flag == False:
                    translated_img = self.translate_image(img_warped, tx=30, ty=0)
                
                elif self.tunnel_static_half_flag == True:
                    translated_img = img_warped

            
                _, L, _ = cv2.split(cv2.cvtColor(translated_img, cv2.COLOR_BGR2HLS))
                _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY) #color_filtering 된 이미지를 한번 더 이진화 하여 차선 검출의 신뢰도를 높였습니다. 

                img_masked = region_of_interest(img_binary) #이진화까지 마친 이미지에 roi를 다시 설정하여줍니다.
                # cv2.imshow("img_masked", img_masked)
                out_img, self.x_location, self.current_lane = self.slidewindow.slidewindow(img_masked, self.is_detected)
                if self.x_location is None:
                    self.x_location = self.last_x_location
                else:
                    self.last_x_location = self.x_location


                img_masked_colored = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
            
                if out_img.shape == img_masked_colored.shape:
                    img_blended = cv2.addWeighted(out_img, 1, img_masked_colored, 0.6, 0)  # sliding window 결과를 시각화하기 위해 out_img와 시점 변환된 이미지를 merging 하였습니다.
                    cv2.imshow('img_blended', img_blended)
                else:
                    print(f"Shape mismatch: out_img {out_img.shape}, img_masked {img_masked_colored.shape}")
                
                self.motor_msg = 5
                angle = pid.pid_control(self.x_location - 320)


                # ---------------------- 정적 회피 주행 ---------------------- # 
                if self.is_dynamic == False:
                    tunnel_statlc_roi_check_arr = [0, 0]
                    self.theta_list = []
                    # 터널 정적 PE-Drum s자 회피
                    if len(self.obstacle_list) > 0:
                        last_tunnel_static_obstacle = None

                        for obstacle in self.obstacle_list:

                            if self.tunnel_static_half_flag == False:
                                # 전방 체크
                                if 0.5 < obstacle.x < 4.5 and -2 <= obstacle.y <= 2:
                                    tunnel_statlc_roi_check_arr[1] = 1
                                # 왼쪽뒤 체크
                                if -6 <= obstacle.x <= -0.5 and 0 < obstacle.y <= 3.5:
                                    tunnel_statlc_roi_check_arr[0] = 1

                                if tunnel_statlc_roi_check_arr == [1, 1]:
                                    self.tunnel_static_half_flag = True
                                    distance_between_obstacle = obstacle.distance()
                        

                            if self.tunnel_static_half_flag == True:
                                
                                if (-0.7 <= obstacle.x <= 5.0) and (-3.0 <= obstacle.y <= 0.5):
                                    last_tunnel_static_obstacle = obstacle

                                    #장애물과 라이다의 상대각도 계산
                                    theta = math.degrees(math.atan2(last_tunnel_static_obstacle.y, last_tunnel_static_obstacle.x))
                                    self.theta_list.append(theta)
                                    print("Theta: ", theta)
                        
                    # 터널 내 장애물 회피 각도 조정
                    for theta in self.theta_list:
                        # 첫번째 장애물과 두번째 장애물을 분리 하려는 조건문: 두번째것만 보기 위함.

                        # 6미터 -100 < theta < 5:

                        # 4미터

                        if theta < -100:
                            break
                        
                        if -100 < theta < 5:

                            # flag를 한번 더 만들기 - 이 때부터는 바로 꺾음
                            if self.tunnel_dynamic_check_flag == False:
                                self.tunnel_dynamic_check_flag = True


                            # 6미터 되는 함수
                            # angle = 0.0027 * (theta**2) - 0.25 * theta - 28.2
                            # 4미터 되는 함수 
                            # angle = 0.0027 * (theta**2) - 0.29 * theta - 28.2
                            

                            # fmtc에서는 됐던 값.
                            # angle = 0.0028 * (theta**2) - 0.42 * theta - 30.0

                            angle = 0.0024 * (theta**2) - 0.4 * theta - 30.0
                            break
                # ---------------------- 정적 회피 주행 ---------------------- # 







                # 최종 계산된 조향각의 변화량을 비교하기 때문에 이 동적 판단 코드는 이 위치가 맞음.
                # 매번 조향각 추가
                self.tunnel_dynamic_steering_list.append(angle)
                if len(self.tunnel_dynamic_steering_list) > self.tunnel_dynamic_steering_list_max_len:
                    self.tunnel_dynamic_steering_list.pop(0)

                
                # print('self.tunnel_dynamic_steering_list', self.tunnel_dynamic_steering_list)
                # len == 5
                for i in range(len(self.tunnel_dynamic_steering_list) - 1): # 0 ~ 3
                    self.steer_diff = abs(self.tunnel_dynamic_steering_list[-1] - self.tunnel_dynamic_steering_list[i])
                    print(f"조향각 변화량[{i}]: {self.steer_diff}")
                    if self.steer_diff > self.steer_diff_threshold:
                        self.is_steer_stable = False
                        break
                else:
                    self.is_steer_stable = True


                print('정적 미션 중간: ', self.tunnel_static_half_flag)
                print('정적 조향 시작?: ', self.tunnel_dynamic_check_flag)
                print('스티어링 안정함?: ', self.is_steer_stable)
                print('동적 미션 플래그: ', self.is_dynamic)



                # 동적 flag 올리기 위함
                # self.tunnel_dynamic_check_flag + steering(변화량이 적을때)  -> is_dynamic(True)
                if (self.tunnel_dynamic_check_flag == True) and (self.is_steer_stable == True):
                    self.is_dynamic = True


                # 정적 상황이 끝난 뒤부터는 계속 들어감
                if self.is_dynamic == True:
                    
                    # Obstacle 이 있을 때 서행 (2~3 정도로 엄청 느리게)
                    if len(self.obstacle_list) > 0:
                        for obstacle in self.obstacle_list:
                            if (0.0 < obstacle.x < 7.0) and (-4.0 < obstacle.y < 3.2):
                                self.motor_msg = 5

                                # 영역 안 장애물의 y 리스트를 갱신
                                self.tunnel_dynamic_obstacle_y_list.append(obstacle.y)
                                if len(self.tunnel_dynamic_obstacle_y_list) > self.tunnel_dynamic_obstacle_y_list_max_len:
                                    self.tunnel_dynamic_obstacle_y_list.pop(0)
                    else:
                        self.tunnel_dynamic_obstacle_y_list = []

                    # Obstacle 이 움직일 때 멈추기
                    if len(self.tunnel_dynamic_obstacle_y_list) >= 2:
                        # print("동적 Y좌표 차이 : ", self.obstacle_y_diff)
                        self.obstacle_y_diff = abs(self.tunnel_dynamic_obstacle_y_list[1] - self.tunnel_dynamic_obstacle_y_list[-2])
                        if self.obstacle_y_diff > self.obstacle_y_diff_threshold:
                            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                            self.publishCtrlCmd(0.0, angle, 1.0)
                            continue


                print()
                
                # 데이터 저장
                self.save_data(img_warped, self.x_location)
                print(2)
                self.publishCtrlCmd(self.motor_msg, angle, 0)

                cv2.waitKey(1)

            except Exception as e:
                print(e)

            rate.sleep()

    # def image_callback(self, msg):
    #     self.image = self.bridge.compressed_imgmsg_to_cv2(msg)
    def init_csv(self):
        try:
            with open(self.csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['image_path', 'x_location'])
                print("CSV 파일 초기화 성공: ", self.csv_path)
        except Exception as e:
            print("CSV 파일 초기화 오류: ", e)

    def save_data(self, img, x_location):
        try:
            timestamp = int(time.time() * 1000)
            image_filename = f'image_{timestamp}.png'
            image_path = os.path.join(self.data_dir, image_filename)
            cv2.imwrite(image_path, img)
            print("이미지 저장 성공: ", image_path)

            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([image_path, x_location])
                print("데이터 저장 성공: ", self.csv_path)
        except Exception as e:
            print("데이터 저장 오류: ", e)


    def objectCB(self, msg):
        self.obstacle_list = []
        for marker in msg.markers:
            obstacle = Obstacle(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list.append(obstacle)
        
        # 장애물을 거리순으로 정렬
        self.obstacle_list.sort(key=lambda obstacle: obstacle.distance())

    def camCB(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def translate_image(self, image, tx, ty):
        """
        이미지 평행이동 함수
        :param image: 입력 이미지
        :param tx: x축 평행이동 거리
        :param ty: y축 평행이동 거리
        :return: 평행이동된 이미지
        """
        rows, cols = image.shape[:2]
        
        # 평행이동 행렬 생성
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 이미지 평행이동
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        
        return translated_image

    def inference(self, gt_img_org):
        # BGR 순서
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        gt_image = gt_image.to(self.device)
        # lane segmentation 
        binary_final_logits, instance_embedding = self.LaneNet_model(gt_image.unsqueeze(0))
        binary_final_logits, instance_embedding = binary_final_logits.to('cpu'), instance_embedding.to('cpu') 
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        binary_img[0:65,:] = 0 #(0~85행)을 무시 - 불필요한 영역 제거      
        binary_img=binary_img.astype(np.uint8)
        binary_img[binary_img>0]=255
        # 차선 클러스터링, 색상 지정
        # rbg_emb, cluster_result = process_instance_embedding(instance_embedding, binary_img,distance=1.5, lane_num=2)
        # rbg_emb = cv2.resize(rbg_emb, dsize=(org_shape[1], org_shape[0]), interpolation=cv2.INTER_LINEAR)
        # a = 0.1
        # frame = a * gt_img_org[..., ::-1] / 255 + rbg_emb * (1 - a)
        # frame = np.rint(frame * 255)
        # frame = frame.astype(np.uint8)

        frame = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, dsize=(640, 480))

        return frame
    
    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.drive_value.throttle = int(motor_msg)
        self.drive_value.steering = servo_msg
        self.drive_value.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.drive_value)

if __name__ == "__main__":
    try:
        lanenet_detection_node = LanenetDetection()
    except rospy.ROSInterruptException:
        pass