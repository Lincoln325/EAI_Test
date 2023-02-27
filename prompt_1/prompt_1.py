import time
import random
from functools import partial

from pytube import YouTube
import cv2 as cv
import numpy as np


class Prompt_1:
    def __init__(self):
        yt = YouTube('https://www.youtube.com/watch?v=6hyLdfYIcxI&ab_channel=WildlifeKingdom')
        self.video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()

        self.roihist = np.ones((256,1))
        self.roihist_weight_step = 0.05
        self.roihist_len = 0
        self.roihist_maxlen = 10
    
    def update_roihist(self, new_roihist):
        cv.normalize(new_roihist,new_roihist,0,255,cv.NORM_MINMAX)
        if self.roihist_len > self.roihist_maxlen:
            pass
        else:
            weight = np.exp(-(self.roihist_weight_step*self.roihist_len))
            self.roihist = (1-weight)*self.roihist + weight*(new_roihist)
            cv.normalize(self.roihist,self.roihist,0,255,cv.NORM_MINMAX)
            self.roihist_len += 1

    
    def area_extract(self, filter_contour, foreground):
        # Contour detection
        contours, _ = cv.findContours(foreground, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
        contours = list(filter(filter_contour, contours))

        # Convex hull generation
        hulls = []
        for i in range(len(contours)):
            hull = cv.convexHull(contours[i])
            hulls.append(hull)

        # Mask creation
        mask = np.zeros((foreground.shape[0], foreground.shape[1], 1), dtype=np.uint8)
        cv.drawContours(mask, hulls, None, color=(255, 255, 255), thickness=cv.FILLED)

        return mask, contours, hulls
    
    @staticmethod
    def filter_contour(contour, threshold):
        return cv.contourArea(contour) > threshold

    def main(self):
        frames = cv.VideoCapture(self.video)

        if (frames.isOpened()== False):
            raise Exception("Error in opening video!")

        previous_frame = None
        start = time.time()

        filter_contour_1 = partial(self.filter_contour, threshold=10000)
        filter_contour_2 = partial(self.filter_contour, threshold=1500)

        while(frames.isOpened()):

            ret, frame = frames.read()
            if ret == True:

                # Resize frame
                frame = cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

                if previous_frame is None:
                    previous_frame = frame.copy()
                    continue

                current_frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                previous_frame_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)  

                # Compute frame difference
                frame_diff = cv.absdiff(current_frame_gray,previous_frame_gray)
                previous_frame = frame.copy()

                # Scene change detection
                if np.sum(frame_diff > 0) > np.multiply(frame.shape[0], frame.shape[1])*0.95:
                    print(time.time() - start)
                    self.roihist_len = 0
                    continue
                
                # Foreground detection
                foreground = cv.morphologyEx(
                    ((frame_diff>50)*255).astype(np.uint8), 
                    cv.MORPH_CLOSE, 
                    cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)), 
                    iterations=3)

                # ROI extraction
                mask, contours, hulls = self.area_extract(filter_contour_1, foreground)

                # Histogram calculation of ROI
                frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

                channels = [0]
                ranges = [-110,30]

                roihist = cv.calcHist([frame_hsv], channels, mask, [256], ranges=ranges)
                self.update_roihist(roihist)

                # Back projection using the detected ROI
                dst = cv.calcBackProject([frame_hsv],channels,self.roihist, ranges=ranges, scale=1)
                dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9)), iterations=1)
                dst = cv.morphologyEx(dst, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)

                # New ROI detection using back projection result
                mask, contours, hulls = self.area_extract(filter_contour_2, dst)

                # Draw the new ROI
                drawing = frame.copy()
                for i in range(len(contours)):
                    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
                    cv.drawContours(drawing, hulls, i, color)
                
                # Center points of the detected butterflies
                centers = []
                for c in contours:
                    M = cv.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv.circle(drawing, (cX, cY), 10, (0, 0, 255), -1)
                    centers.append(M)
                
                # Display number of detected butterflies
                cv.putText(drawing, str(len(centers)), (10, 40), cv.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3, cv.LINE_AA)

                cv.imshow("drawing", drawing)
            
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        frames.release()

        cv.destroyAllWindows()
        cv.waitKey(1)


if __name__ == "__main__":
    prompt_1 = Prompt_1()
    prompt_1.main()