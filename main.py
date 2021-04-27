import math
import pickle

import cv2
import cv2 as cv
import numpy as np
import time as t
from PIL import Image

fps = 0
fps_avg = 0
fps_acc = 0
start_t = t.perf_counter()
time = t.perf_counter()


def find_image_targets(img, target, threshold=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF_NORMED)
    return np.where(res >= threshold)


def squares_on_img(img, targets, w, h):
    for pt in zip(*targets[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


def circle_on_img(img, x, y, r):
    cv2.circle(img, (x, y), r, (0, 255, 0), 2)


def find_diff(img1, img2):
    diff = cv2.absdiff(img1, img2)
    # conv_hsv_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(conv_hsv_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # diff[mask != 255] = [255, 255, 255]
    # cv.imshow('output/sovs.png', mask)
    return diff


def find_boxes_from_img(img, output):
    threshold_area = 500
    threshold_area_max = 5000
    ret, thresh_gray = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 15, 255, cv.THRESH_BINARY)
    contours, hier = cv.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cont in contours:
        area = cv.contourArea(cont)

        if area < threshold_area:
            cv.fillPoly(thresh_gray, pts=[cont], color=0)
            continue

        rect = cv.minAreaRect(cont)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)
        if (aspect_ratio > 2.5):  # or w > 100 or h > 100:
            cv2.fillPoly(thresh_gray, pts=[cont], color=0)
            continue

    thresh_gray2 = cv2.morphologyEx(thresh_gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
    contours, hier = cv.findContours(thresh_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for cont in contours:
        area = cv.contourArea(cont)
        if area < 750 or area > threshold_area_max:
            cv2.fillPoly(thresh_gray2, pts=[cont], color=0)
            continue
        rect = cv.minAreaRect(cont)
        (x, y), (w, h), _ = rect
        # if area <= 2000:
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(output, [box], 0, (0, 255, 0), 1)
        # else:
        #     dst_points = [(0, 0), (0, int(h)), (int(w), int(h)), (int(w), 0)]
        #     src_points = [(x - int(w), y - int(h)), (x - int(w), y + int(h)), (x + int(w), y + int(h)),
        #                   (x + int(w), y - int(h))]
        #     for cord in create_homography_and_wrap(
        #             cv.cvtColor(thresh_gray, cv.COLOR_GRAY2RGB),
        #             np.float32(src_points).reshape(-1, 1, 2),
        #             np.float32(dst_points).reshape(-1, 1, 2),
        #             int(h),
        #             int(w)
        #     ):
        #         output = cv2.circle(output, (cord[0], cord[1]), cord[2], (0, 0, 255), 4)
    output = cv.putText(output, "FPS: " + str(fps) + ", AVG: " + str(fps_avg), (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 255, 255))

    #backtorgb1 = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2RGB)
    #backtorgb = cv2.cvtColor(thresh_gray2, cv2.COLOR_GRAY2RGB)
    ### SLOW IMG SHOW FOR DEBUGGING :D
    #cv.imshow('output/sovs.png', np.vstack([np.hstack([img, backtorgb1]), np.hstack([backtorgb, output])]))
    ### FAST IMG SHOW FOR SHOWING OFF :P
    cv.imshow('output/sovs.png', output)


def create_homography_and_wrap(img, src_points, dest_points, height, width):
    M, mask = cv2.findHomography(src_points, dest_points)
    M_back, _ = cv2.findHomography(dest_points, src_points)

    warpedImg = cv2.warpPerspective(img, M, (width, height))
    warpedImg_g = cv2.cvtColor(warpedImg, cv2.COLOR_BGR2GRAY)
    warpedImg_g = cv.blur(warpedImg_g, (3, 3))
    circles = cv2.HoughCircles(warpedImg_g, cv2.HOUGH_GRADIENT, 2,
                               5, param1=100, param2=20, minRadius=7, maxRadius=15)
    circle_points = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            print(r)
            p_origin_homogenous = np.array((x, y, 1)).reshape(3, 1)
            temp_p = M_back.dot(p_origin_homogenous)
            sum = np.sum(temp_p, 1)
            px = int(round(sum[0] / sum[2]))
            py = int(round(sum[1] / sum[2]))
            circle_points.append((px, py, r))
    return circle_points


# filename = 'test_images/mt.mp4'
# filename2 = 'test_images/mtc.png'
# template_name = 'templates/tt'
# img = cv2.imread(filename)
# img2 = cv2.imread(filename2)
# img2 = img2[99:632, 119:1152]
# src_points = []
# width, height = (1024, 512)
# dst_points = np.float32([(0,0), (width,0), (width,height), (0,height)]).reshape(-1,1,2)
# for i in range(1, 5):
#    template = cv2.imread(template_name + str(i) + '.png', 0)
#    w, h = template.shape[::-1]
#    targets = find_image_targets(img, template, 0.65)
#    src_points.append(list(zip(*targets[::-1]))[0])
#    squares_on_img(img, targets, w, h)
# src_points.append(src_points.pop(0))
# src_points.append(src_points.pop(0))
# src_points.append(src_points.pop(0))
# M, mask = cv2.findHomography(np.float32(src_points).reshape(-1,1,2), dst_points, cv2.RANSAC,5.0)
# im_dst = cv2.warpPerspective(img, M, (width, height))

# Load calib values
pik = pickle.load(open("undist_params.p", "rb"))
mtx = pik["mtx"]
dist = pik["dist"]

cap = cv2.VideoCapture(1)
ret, frame = cap.read()
img2 = frame
img2 = cv2.undistort(img2, mtx, dist, None, mtx)

while ret:
    # cv2.imshow("sovs", frame)
    cv2.waitKey(1)
    diff_img = find_diff(frame, img2)
    # frame = cv.putText(frame, "FPS: " + str(fps), (0,500), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255))
    # cv2.imshow("sovs", frame)
    find_boxes_from_img(diff_img, frame)
    ret, frame = cap.read()
    frame = cv2.undistort(frame, mtx, dist, None, mtx)

    fps = int(1 / float((t.perf_counter() - time)))
    fps_acc += 1
    fps_avg = int(fps_acc / (t.perf_counter() - start_t))

    time = t.perf_counter()

# cv2.imwrite('output/res3.png', diff_img)
# cv2.imwrite("output/res2.png", im_dst)
# cv2.imwrite('output/res.png', img)
