import numpy as np
import cv2
import sys
import time
import pdb

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # vidcap = cv2.VideoCapture('Experiments/EXP_2_ROBOT_VIDEO_MICRO_CAM/EXP_15.mp4')
    vidcap = cv2.VideoCapture('hough_line_sample.mp4')

    # Exit if video not opened.
    if not vidcap.isOpened():
        print("Could not open video")
        sys.exit()
    try:
        angDArray=np.empty((1))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10.0, (640,480))
        # Read until video is completed
        while(vidcap.isOpened()):
            time.sleep(0)
            # Start timer
            timer = cv2.getTickCount()
            
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            
            # Capture frame-by-frame
            success, image = vidcap.read()
            if not success:
                print('Cannot read video file')
                sys.exit()
            
            # Prepare crop area
            width_percent, height_percent = 0.31, 0.6
            h, w, c = image.shape
            width, height = int(w * width_percent), int(h * height_percent)
            x, y = int(w * (1 - width_percent)/2), int(0.2 * height)
            
            # Crop image to specified area using slicing
            image1 = image[y:y+height, x:x+width]
            cv2.imshow("im", image1)
            # find the edges
            # 1. to gray
            gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            cv2.imshow("gray", gray)
            # 2. to binary image
            ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            cv2.imshow("thresh1", thresh1)
            
            image2 = cv2.Canny(image=thresh1, threshold1=100, threshold2=200,apertureSize = 3)
            
            cv2.imshow("Croppeed Image", image1)
            cv2.imshow("Canny Image", image2)
            # 4. thicken the edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            edges = cv2.dilate(image2, kernel)
            cv2.imshow("edges2", image2)

            # Display FPS on frame
            cv2.putText(image1, "FPS : " + str(int(fps)), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            numlines = 4
            angles = []
            lines = cv2.HoughLinesP(image2, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            x1_ref, y1_ref = 121, 306
            x2_ref, y2_ref = 184, 308
            
            m_ref = (y2_ref-y1_ref)/(x2_ref-x1_ref)
            
            ref_line_in_pt_x  = np.array([0,500])
            ref_line_end_pt_y = y1_ref+m_ref*(ref_line_in_pt_x-x1_ref)
            
            # ref_line_in_pt_y=y1_ref+m_ref*(ref_line_in_pt_x-x1_ref)
            # ref_line_end_pt_y=y1_ref+m_ref*(ref_line_in_pt_x-x1_ref)
            
            cv2.line(image1, (ref_line_in_pt_x[0], int(ref_line_end_pt_y[0])), 
                (ref_line_in_pt_x[1] , int(ref_line_end_pt_y[1])), (255, 255, 0), 10)
            
            cv2.line(image1, (51, 183), 
                (154 , 192), (255, 255, 0), 2)
            
            for k in range(numlines):
                for x1, y1, x2, y2 in lines[k]:
                    # cv2.line(image1, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    
                    # cv2.circle(image1, (x1, y1), 5, (0,255,0), 2)
                    # cv2.circle(image1, (x2, y2), 5, (0,255,0), 2)

                    #Determine slope of each line
                    angles.append(np.arctan2(y2-y1,x2-x1))
                    
            # print(abs(angles[1]-angles[0])*180.0/np.pi)
            # angDArray=np.vstack((angDArray,abs(angles[1]-angles[0])*180.0/np.pi))
                #Find the maximum angle among all the angles and find its corresponding lines
            angD = np.zeros(((numlines)*(numlines)-(numlines),3))
            k = 0
            for i in range(numlines):
                for j in range(numlines):
                    if j!=i:
                        angD[k,:] = np.array((i,j,abs(angles[j]-angles[i])*180.0/np.pi))
                        k = k+1                        
            max_index_col = np.argmax(angD[:,2], axis=0)
            
            angDArray=np.vstack((angDArray,np.max(angD[:,2])))
            
            #Draw green lines on the maximum angle
            for k in range(2):
                for x1, y1, x2, y2 in lines[int(angD[max_index_col,:][k])]:
                    try:
                        print(x1,y1,x2,y2)
                        
                        if int((y2-y1)/(x2-x1) - m_ref) != 0:
                            m=(y2-y1)/(x2-x1)
                            cable_line_end_pt_y = y1+m*(ref_line_in_pt_x-x1)
                            
                            cv2.line(image1, (ref_line_in_pt_x[0], int(cable_line_end_pt_y[0])), 
                                     (ref_line_in_pt_x[1] , int(cable_line_end_pt_y[1])), (255, 255, 0), 2)
                        
                            # cv2.line(image1, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    
                            # cv2.circle(image1, (x1, y1), 5, (0,255,0), 2)
                            # cv2.circle(image1, (x2, y2), 5, (0,255,0), 2)
                    except:
                        print('Line cant be determined cause (y2-y1)/(x2-x1) tends to inf')
                            
            print('---------')
            # print(np.max(angD[:,2]), max_index_col)
            
            
    
            cv2.imshow("Final image with Lines", image1)
            # cv2.putText(image1,str(angD),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255),2)
            
            out.write(image1)
            # if angD<1:
            #     pdb.set_trace()
            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        # When everything done, release the video capture object
        vidcap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        
    finally:
        # When everything done, release the video capture object
        vidcap.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        
        plt.plot(angDArray)
    

