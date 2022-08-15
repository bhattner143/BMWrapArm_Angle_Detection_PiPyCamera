#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 15:07:30 2022

@author: dips-pi
"""

import numpy as np
import cv2
import sys
import time
import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # vidcap = cv2.VideoCapture('Experiments/EXP_2_ROBOT_VIDEO_MICRO_CAM/EXP_15.mp4')
    # vidcap = cv2.VideoCapture('hough_line_sample.mp4')
    cam_num_list = [0]
    df        = pd.DataFrame()
    for cam_num in cam_num_list:
        
        cap = cv2.VideoCapture(cam_num)
        cap.set(cv2.CAP_PROP_EXPOSURE, 50000)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")   
        try:
            angDArray=np.empty((1))
            # out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10.0, (640,480))
            # Read until video is completed
            while(cap.isOpened()):
                time.sleep(0)
                # Start timer
                timer = cv2.getTickCount()
                
                # Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                
                # Capture frame-by-frame
                success, image = cap.read()
                if not success:
                    print('Cannot read video file')
                    sys.exit()
                
                # Display the resulting frame
                # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                '''
                width_percent, height_percent = 1, 0.3
                h, w, c = image.shape
                width, height = int(w * width_percent), int(h * height_percent)
                x = int(w * (1 - width_percent)/2)
                y =  int(0.7 * height)
                ''' 
                
                """Crop image to specified area using slicing"""
                #image1 = image[y:y+height, x:x+width]
                h, w, c = image.shape
                image1 = image[int(0.43*h):int(0.7*h), int(0.43*w):int(0.65*w)]
                cropped = image1
                cv2.imshow("Cropped Image", cropped)
                
                """Display the results in binary"""
                # 1. to gray
                #gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
                gray = image1[:,:,1]
                cv2.imshow("Gray", gray)
                
                # 2. to binary image
                ret, thresh1 = cv2.threshold(gray, 120, 180, cv2.THRESH_BINARY)
                
                cv2.imshow("Binary", thresh1)
                
                """Detect the edges of the binary image by using a Canny detector"""
                image2 = cv2.Canny(image=thresh1, threshold1=100, threshold2=200,apertureSize = 3)
                cv2.imshow("Canny Image", image2)
                
                """Thicken the edges"""
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                edges = cv2.dilate(image2, kernel)
                cv2.imshow("Edges", edges)
                '''
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue
                '''
                """ Display FPS on frame"""
                cv2.putText(image1, "FPS : " + str(int(fps)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
                numlines = 4 # Num lines to be detected
                angles = []

                if cam_num == 1 or cam_num == 7:
                    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=150)
                elif cam_num>1 and cam_num<7:
                    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=180)
                else:
                    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=80)
                
                ver_lines_theta = []
                hor_lines_theta = []
                ver_lines_rho = []
                hor_lines_rho = []
                
                """For no detected lines"""
                if lines is None:
                    continue
                
                """Loop through all the detected lines to find line moving 
                   through the cable"""
                print('\n')
                angle_threshold   = 90
                ver_theta_initial = 30
                ver_theta         = ver_theta_initial
                
                for k in range(len(lines)):
                    
                    # Taking the upper bound of angle from previous recorded theta
                    angle_threshold = ver_theta + 40
                    
                    for line in lines[k]:
                        rho = line[0]
                        theta = line[1]
                        
                        # calc the angle
                        angle = theta % np.pi
        
                        # find vertical lines
                        if angle/np.pi*180 < angle_threshold:#np.pi / 2:
                            
                            # draw the line
                            pt1 = (int(rho / np.cos(theta)), 0)
                            pt2 = (int((rho - image1.shape[0] * np.sin(theta)) / np.cos(theta)), image1.shape[0])
                            
                            
                            cv2.line(image1, pt1, pt2, (0, 255, 0), 2)
                            # append to the list
                            # if angle/np.pi*180 < 80:
                            print('Actual vertical angle array',angle/np.pi*180)
                            ver_lines_theta.append(angle)
                            ver_lines_rho.append(abs(rho))
                        else:
                            # draw the line
                            # print('dipu', rho, theta) 
                            pt1 = (0, int(rho / np.sin(theta)))
                            pt2 = (image.shape[1], int((rho - image.shape[1] * np.cos(theta)) / np.sin(theta)))
                            cv2.line(image, pt1, pt2, (255, 255, 0), 2)
                            # append to the list
                            hor_lines_theta.append(angle)
                            print('Actual horizontal angle array',angle/np.pi*180)
                            hor_lines_rho.append(abs(rho))
                            
                        if not len(hor_lines_theta) or not len(ver_lines_theta):
                            continue
                        
                        """ """
                        """draw the angle vertical line"""        
                        ver_lines_theta_array = np.array([])
                        
                        # find out the outlier line
                        ver_lines_theta_array = np.array(ver_lines_theta)
                        ver_lines_rho_array_thresholded   = np.array(ver_lines_rho)
                        c = np.array(ver_lines_rho)
                        
                        d = np.abs(ver_lines_theta_array- np.median(ver_lines_theta_array))
                        mdev = np.median(d)
                        s = d/(mdev if mdev else 1)
                        
                        # Remove the outlier line by adaptive thresholding
                        threshold = 0.1
                        ver_lines_theta_array_thresholded = np.array([])
                        
                        while ver_lines_theta_array_thresholded.shape[0] == 0:
                            ver_lines_theta_array_thresholded = ver_lines_theta_array[s<threshold]
                            threshold = threshold + 0.1
                            
                        np.argwhere(s<threshold)
                        ver_lines_rho_array_thresholded[np.argwhere(s<threshold)]
                        
                        print('After thresholding',ver_lines_theta_array_thresholded/np.pi*180)
                        
                        ver_lines_rho_array_to_list = ver_lines_rho_array_thresholded.tolist()
                        ver_lines_theta_array_to_list = ver_lines_theta_array_thresholded.tolist()
                        
                        ver_rho = sum(ver_lines_rho_array_to_list) / len(ver_lines_rho_array_to_list)
                        ver_theta = sum(ver_lines_theta_array_to_list) / len(ver_lines_theta_array_to_list)
                        
                        """draw the average vertical line"""
                        pt1 = (0, int(ver_rho / np.sin(ver_theta)))
                        pt2 = (image1.shape[1], int((ver_rho - image1.shape[1] * np.cos(ver_theta)) / np.sin(ver_theta)))
                        cv2.line(image1, pt1, pt2, (0, 0, 255), 5)
                        
                        """to degree"""
                        # hor_theta_ref = hor_theta_ref / np.pi * 180
                        ver_theta     = ver_theta / np.pi * 180
                        
                        """print to console"""
                        angle_determined = ver_theta#hor_theta_ref  - ver_theta
                        # print("The angle is:" + str(angle_determined))
                        angDArray = np.vstack((angDArray,angle_determined))
                        
                        cv2.imshow("Final image with Lines", image1)
                        image2 = cv2.putText(image1,str(round(angle_determined,2)),
                                    (200,700), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2,cv2.LINE_AA)
                        # Displaying the image
                        cv2.imshow('final Video', image2) 
                
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            # When everything done, release the video capture object
            cap.release()
            # Closes all the frames
            cv2.destroyAllWindows() 
            print("\nGlobal quitting")
            break
        
        finally:
            # When everything done, release the video capture object
            
            cap.release()
            # Closes all the frames
            cv2.destroyAllWindows()
        