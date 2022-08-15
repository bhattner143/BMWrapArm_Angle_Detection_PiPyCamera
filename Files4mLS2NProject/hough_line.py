import numpy as np
import cv2

vidcap = cv2.VideoCapture('hough_line_sample.mp4')
success, image = vidcap.read()
# image = cv2.imread('hough_lin_sample.mp4')

# Prepare crop area
width_percent, height_percent = 0.4, 0.7
h, w, c = image.shape
width, height = int(w * width_percent), int(h * height_percent)
x, y = int(w * (1 - width_percent)/2), int(0.01 * height)
# Crop image to specified area using slicing
image = image[y:y+height, x:x+width]
cv2.imshow("Crop", image)
cv2.waitKey(0)

image = cv2.Canny(image=image, threshold1=100, threshold2=200)
cv2.imshow("Canny", image)
cv2.waitKey(0)

angles = []
lines = cv2.HoughLinesP(image, 1, np.pi/180, 10, minLineLength=10, maxLineGap=10)
for k in range(2):
    for x1, y1, x2, y2 in lines[k]:
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
        angles.append(np.arctan2(y2-y1,x2-x1))
print(abs(angles[1]-angles[0])*180.0/np.pi)
cv2.imshow("line", image)
cv2.waitKey(0)
