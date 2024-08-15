#!/usr/bin/python3
#-*- encoding: utf-8 -*-
import os.path as ops
import numpy as np
import torch
import cv2
import time
import math
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
# from race.msg import drive_values
from driving_es import *
from slidewindow_morai import *

from lidar_object_detection.msg import ObjectInfo
from visualization_msgs.msg import MarkerArray

from morai_msgs.msg import GetTrafficLightStatus, CtrlCmd, GPSMessage

import csv






class Obstacle:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def distance(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error

class LanenetDetection:
    def __init__(self):
        rospy.init_node('lanenet_detection_node')

        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.image_callback)
        rospy.Subscriber("/bounding_box", MarkerArray, self.objectCB)

        self.ctrl_cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=1)
        self.ctrl_cmd_msg = CtrlCmd()
        self.ctrl_cmd_msg.longlCmdType = 2

        self.bridge = CvBridge()
        self.image = np.empty(shape=[0])
        self.slidewindow = SlideWindow()
        self.x_location = 320
        self.last_x_location = 320
        self.is_detected = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = '/home/seongdonghyun/iscc_2024/src/camera/lanenet/src/lanenet_.model'
        self.LaneNet_model = Lanenet(2, 4)
        self.LaneNet_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        self.LaneNet_model.to(self.device)

        # 데이터 저장 경로 및 CSV 파일 설정
        self.data_dir = '/home/seongdonghyun/iscc_2024/data'
        os.makedirs(self.data_dir, exist_ok=True)
        self.csv_path = os.path.join(self.data_dir, 'lane_data.csv')
        self.init_csv()

        # 동적 장애물 파라미터
        self.tunnel_dynamic_check_flag = False
        self.is_dynamic = False
        self.obstacle_list = []
        self.theta_list = []
        self.tunnel_dynamic_steering_list = []
        self.tunnel_dynamic_steering_list_max_len = 20
        self.steer_diff = -1
        self.steer_diff_threshold = 5
        self.is_steer_stable = False
        self.tunnel_dynamic_obstacle_y_list = []
        self.tunnel_dynamic_obstacle_y_list_max_len = 50
        self.obstacle_y_diff = -1
        self.obstacle_y_diff_threshold = 0.2

        # 정적 장애물 파라미터
        self.tunnel_static_half_flag = False

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                pid = PID(0.015, 0.003, 0.010)
                cv2.imshow("lane_image", self.image)

                show_img = self.inference(self.image)
                cv2.imshow("show", show_img)

                img_frame = show_img.copy()
                img_roi = img_frame[280:470, 0:]

                img_filtered = color_filter(img_roi)
                cv2.imshow("filter", img_filtered)

                left_margin = 195
                top_margin = 44
                src_points = np.float32([[0, 190], [left_margin, top_margin], [img_filtered.shape[1]-left_margin, top_margin], [img_filtered.shape[1], 190]])
                dst_points = np.float32([[img_filtered.shape[1]//4, 190], [img_filtered.shape[1]//4, 0], [img_filtered.shape[1]//4*3, 0], [img_filtered.shape[1]//4*3, 190]])

                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                img_warped = cv2.warpPerspective(img_filtered, matrix, (img_filtered.shape[1], img_filtered.shape[0]))

                img_warped = cv2.resize(img_warped, dsize=(640, 480))
                cv2.imshow('img_warped', img_warped)

                translated_img = img_warped
                _, L, _ = cv2.split(cv2.cvtColor(translated_img, cv2.COLOR_BGR2HLS))
                _, img_binary = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY)

                img_masked = region_of_interest(img_binary)
                out_img, self.x_location, _ = self.slidewindow.slidewindow(img_masked, self.is_detected)

                if self.x_location is None:
                    self.x_location = self.last_x_location
                else:
                    self.last_x_location = self.x_location

                img_masked_colored = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
                if out_img.shape == img_masked_colored.shape:
                    img_blended = cv2.addWeighted(out_img, 1, img_masked_colored, 0.6, 0)
                    cv2.imshow('img_blended', img_blended)

                angle = pid.pid_control(self.x_location - 320)
                servo_msg = -radians(angle)
                self.motor_msg = 7

                # 데이터 저장
                self.save_data(img_warped, self.x_location)

                self.publishCtrlCmd(self.motor_msg, servo_msg, 0)
                cv2.waitKey(1)

            except Exception as e:
                print(e)

            rate.sleep()

    def init_csv(self):
        # CSV 파일 초기화 및 헤더 추가
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_path', 'x_location'])

    def save_data(self, img, x_location):
        # 이미지 저장
        timestamp = int(time.time() * 1000)
        image_filename = f'image_{timestamp}.png'
        image_path = os.path.join(self.data_dir, image_filename)
        cv2.imwrite(image_path, img)

        # CSV 파일에 이미지 경로와 x_location 값 기록
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([image_path, x_location])

    def image_callback(self, msg):
        self.image = self.bridge.compressed_imgmsg_to_cv2(msg)

    def objectCB(self, msg):
        self.obstacle_list = []
        for marker in msg.markers:
            obstacle = Obstacle(marker.pose.position.x, marker.pose.position.y, marker.pose.position.z)
            self.obstacle_list.append(obstacle)

        # 장애물을 거리순으로 정렬
        self.obstacle_list.sort(key=lambda obstacle: obstacle.distance())

    def translate_image(self, image, tx, ty):
        rows, cols = image.shape[:2]
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))
        return translated_image

    def inference(self, gt_img_org):
        org_shape = gt_img_org.shape
        gt_image = cv2.resize(gt_img_org, dsize=(512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = gt_image / 127.5 - 1.0
        gt_image = torch.tensor(gt_image, dtype=torch.float)
        gt_image = np.transpose(gt_image, (2, 0, 1))
        gt_image = gt_image.to(self.device)
        binary_final_logits, instance_embedding = self.LaneNet_model(gt_image.unsqueeze(0))
        binary_final_logits, instance_embedding = binary_final_logits.to('cpu'), instance_embedding.to('cpu')
        binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().numpy()
        binary_img[0:65, :] = 0
        binary_img = binary_img.astype(np.uint8)
        binary_img[binary_img > 0] = 255

        frame = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.resize(frame, dsize=(640, 480))

        return frame

    def publishCtrlCmd(self, motor_msg, servo_msg, brake_msg):
        self.ctrl_cmd_msg.velocity = motor_msg
        self.ctrl_cmd_msg.steering = servo_msg
        self.ctrl_cmd_msg.brake = brake_msg
        self.ctrl_cmd_pub.publish(self.ctrl_cmd_msg)

if __name__ == "__main__":
    try:
        lanenet_detection_node = LanenetDetection()
    except rospy.ROSInterruptException:
        pass
