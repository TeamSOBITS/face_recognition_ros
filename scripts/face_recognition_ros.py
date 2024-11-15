#!/usr/bin/env python3
# coding: utf-8
import rospy

from sensor_msgs.msg import Image
from sobits_msgs.msg import BoundingBoxes
from sobits_msgs.msg import BoundingBox
from sobits_msgs.srv import RunCtrl
from sobits_msgs.srv import RunCtrlResponse
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

import os
import face_recognition
from cv_bridge import CvBridge
import cv2
import numpy as np

# import matplotlib.pyplot as plt


class FaceRecognitionServer():
    def __init__(self):
        self.bridge = CvBridge()
        self.labeling_name_picture_path_ = str(rospy.get_param("labeling_directory_path", "/home/sobits/catkin_ws/src/face_recognition_ros/picture_folder"))
        self.detect_ctr_ = rospy.get_param("execute_flag", True)
        self.pub_image_ = rospy.Publisher("/face_recognition_ros/label_result", Image, queue_size=1)
        self.pub_rect_ = rospy.Publisher("/face_recognition_ros/name_rect", BoundingBoxes, queue_size=1)
        start_flag, detect_counter = self.label_picture_input()
        if (start_flag):
            print("\n", detect_counter, "人の顔をセットしました。\n検出を開始します。\n")
            self.sub_ = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback_image)
            rospy.Service("/face_recognition_ros/update_label", Trigger, self.update_labels)
            rospy.Service("/face_recognition_ros/run_ctr", RunCtrl, self.detect_ctr)
            rospy.spin()
        else:
            rospy.logerr("Server ERROR")
            rospy.logerr("That dirctory is not found...")

    def callback_image(self, msg):
        if ((self.detect_ctr_ != True) or (len(self.label_name) == 0)):
            return
        try:
            # ROSのImageメッセージをOpenCVのBGR画像に変換
            bgr_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # BGRからRGBに変換
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            rospy.logerr(f"Failed to process image: {e}")
            return

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Loop through each face in this frame of video
        unknown_counter = 0
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = msg.header
        bounding_boxes.bounding_boxes = []
        for (ymin, xmax, ymax, xmin), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.encodings, face_encoding)

            name = "unknown" + str(unknown_counter)

            # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.label_name[best_match_index]
                # Draw a box around the face
                cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                label = name + " {:.2f}".format(1.0 - face_distances[best_match_index])
            else:
                unknown_counter += 1
                # Draw a box around the face
                cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                label = name + " 0.00"

            (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # ラベルの背景を描画
            cv2.rectangle(rgb_image, (xmin, ymin - label_height), (xmin + label_width, ymin), (255, 255, 255), cv2.FILLED)

            # テキストを描画
            cv2.putText(rgb_image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            bounding_box = BoundingBox()
            bounding_box.Class = name
            if matches[best_match_index]:
                bounding_box.probability = 1.0 - face_distances[best_match_index]
            else:
                bounding_box.probability = 0.0
            bounding_box.xmin = xmin
            bounding_box.xmax = xmax
            bounding_box.ymin = ymin
            bounding_box.ymax = ymax
            bounding_boxes.bounding_boxes += [bounding_box]

        # RGB画像をROSのImageメッセージに変換してパブリッシュ
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
        self.pub_image_.publish(rgb_msg)
        self.pub_rect_.publish(bounding_boxes)


    def label_picture_input(self):
        if (os.path.isdir(self.labeling_name_picture_path_)):
            files_name_list = os.listdir(self.labeling_name_picture_path_)
            label_name_pic = []
            self.label_name = []
            for file_name in files_name_list:
                label_name_pic += [self.labeling_name_picture_path_ + "/" + file_name]
                self.label_name += [file_name.split(".")[0]]
        else:
            self.label_name = []
            self.encodings = []
            return False, 0

        # 学習データの顔画像を読み込む
        self.encodings = []
        for name in label_name_pic:
            try:
                train_img = face_recognition.load_image_file(name)
                encoding = face_recognition.face_encodings(train_img)
                if (len(encoding) == 1):
                    self.encodings += [encoding[0]]
                else:
                    self.label_name = []
                    self.encodings = []
                    return False, 0
            except Exception as e:
                # Print an error message if loading fails
                print(f"Failed to load image {name}: {e}")
                self.label_name = []
                self.encodings = []
                return False, 0
        return True, len(self.label_name)
    
    def update_labels(self, srv):
        s_detect_ctr_ = self.detect_ctr_
        self.detect_ctr_ = False
        is_ok, counter = self.label_picture_input()
        if (is_ok):
            message_txt = "[Success] Set the Face of " + str(counter) + " persons"
        else:
            message_txt = "[Error] Face Recognition can't Set from the target folder"
            rospy.logerr("That dirctory is not found...")
        self.detect_ctr_ = s_detect_ctr_
        return TriggerResponse(success = is_ok, message = message_txt)
    
    def detect_ctr(self, srv):
        self.detect_ctr_ = srv.request
        if (self.detect_ctr_):
            print("detect ON")
        else:
            print("detect OFF")
        return RunCtrlResponse(response = True)


if __name__ == "__main__":
    rospy.init_node("face_recognition_ros")
    frs = FaceRecognitionServer()