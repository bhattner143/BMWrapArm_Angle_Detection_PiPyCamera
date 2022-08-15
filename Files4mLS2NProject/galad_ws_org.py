import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('hough_line_sample.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        width_percent = 0.4
        height_percent = 0.55
        h, w, c = frame.shape
        width, height = int(w * width_percent), int(h * height_percent)
        x, y = int(w * (1 - width_percent) / 2), int(0.01 * height)
        # Crop image to specified area using slicing
        cropped = frame[y:y + height, x:x + width]
        image = cropped
        cv2.imshow("Crop", cropped)
        # find the edges
        # 1. to gray
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        # 2. to binary image
        ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh1", thresh1)
        # 3. find edges
        edges = cv2.Canny(thresh1, 100, 200, 3)
        cv2.imshow("edges", edges)
        # 4. thicken the edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel)
        cv2.imshow("edges2", edges)
        # 5. find the lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            continue

        ver_lines_theta = []
        hor_lines_theta = []
        ver_lines_rho = []
        hor_lines_rho = []
        # for each line
        for k in range(len(lines)):
            # continue
            for line in lines[k]:
                rho = line[0]
                theta = line[1]
                # calc the angle
                angle = theta % np.pi

                # find vertical lines
                if angle < np.pi / 2:
                    # draw the line
                    pt1 = (int(rho / np.cos(theta)), 0)
                    pt2 = (int((rho - image.shape[0] * np.sin(theta)) / np.cos(theta)), image.shape[0])
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)
                    # append to the list
                    ver_lines_theta.append(angle)
                    ver_lines_rho.append(abs(rho))
                # else is horizontal
                else:
                    # continue
                    # draw the line
                    pt1 = (0, int(rho / np.sin(theta)))
                    pt2 = (image.shape[1], int((rho - image.shape[1] * np.cos(theta)) / np.sin(theta)))
                    cv2.line(image, pt1, pt2, (255, 0, 0), 2)
                    # append to the list
                    hor_lines_theta.append(angle)
                    hor_lines_rho.append(abs(rho))
        if not len(hor_lines_theta) or not len(ver_lines_theta):
            continue

        # calc the average
        hor_rho = sum(hor_lines_rho) / len(hor_lines_rho)
        hor_theta = sum(hor_lines_theta) / len(hor_lines_theta)
        # draw the average horizontal line
        pt1 = (0, int(hor_rho / np.sin(hor_theta)))
        pt2 = (image.shape[1], int((hor_rho - image.shape[1] * np.cos(hor_theta)) / np.sin(hor_theta)))
        cv2.line(image, pt1, pt2, (255, 255, 0), 5)

        ver_rho = sum(ver_lines_rho) / len(ver_lines_rho)
        ver_theta = sum(ver_lines_theta) / len(ver_lines_theta)
        # draw the average vertical line
        pt1 = (0, int(ver_rho / np.sin(ver_theta)))
        pt2 = (image.shape[1], int((ver_rho - image.shape[1] * np.cos(ver_theta)) / np.sin(ver_theta)))
        cv2.line(image, pt1, pt2, (0, 255, 255), 5)
        cv2.imshow("image", image)

        # to degree
        hor_theta = hor_theta / np.pi * 180
        ver_theta = ver_theta / np.pi * 180
        # print to console
        print("The angle is: " + str(hor_theta - ver_theta))
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break
    
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
