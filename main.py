import pickle
import threading
import tkinter as tk
import cv2
import cv2 as cv
import numpy as np
import time as t
top = tk.Tk()
fps = 0
fps_avg = 0
fps_acc = 0
start_t = t.perf_counter()
time = t.perf_counter()
debug = True
reset_first_img = False
frame_to_show = 0
finished = False
morph_size = 5
morph_size_var = tk.StringVar()
morph_size_var.set(str(morph_size))

threshold_area = 500
threshold_area_max = 5000
threshold_area_var = tk.StringVar()
threshold_area_max_var = tk.StringVar()
threshold_area_var.set(str(threshold_area))
threshold_area_max_var.set(str(threshold_area_max))

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

    thresh_gray2 = cv2.morphologyEx(thresh_gray, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size)))
    contours, hier = cv.findContours(thresh_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    max_val, max_x, max_y = 0, 0, 0
    contour_cords = []
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
        contour_cords.append(cont)
        val = find_avg_color(output, cont)
        if max_val < val:
            max_val, max_x, max_y = val, int(x), int(y)
    if max_val > 0:
        output = cv.circle(output, (max_x, max_y), 40, (255, 255, 255), 7)

    output = cv.putText(output, "FPS: " + str(fps) + ", AVG: " + str(fps_avg), (0, 500), cv2.FONT_HERSHEY_SIMPLEX, 3,
                        (255, 255, 255))

    if debug:
        backtorgb1 = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2RGB)
        backtorgb = cv2.cvtColor(thresh_gray2, cv2.COLOR_GRAY2RGB)
        return np.vstack([np.hstack([img, backtorgb1]), np.hstack([backtorgb, output])])
    else:
        return output


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


def find_avg_color(frame, contour):
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv.drawContours(mask, contour, -1, 255, -1)
    mean = cv.mean(frame, mask=mask)
    return (mean[0] + mean[1] + mean[2]) / 3


def test(a):
    # Load calib values
    pik = pickle.load(open("undist_params.p", "rb"))
    mtx = pik["mtx"]
    dist = pik["dist"]
    global fps, fps_acc, fps_avg, time, frame_to_show, reset_first_img, finished

    cap = cv2.VideoCapture("test_images/tm.mov")
    ret, frame = cap.read()
    img2 = frame
    frame_to_show = frame
    img2 = cv2.undistort(img2, mtx, dist, None, mtx)
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.undistort(frame, mtx, dist, None, mtx)

        if reset_first_img:
            img2 = frame
            reset_first_img = False
            continue

        diff_img = find_diff(frame, img2)
        frame_to_show = find_boxes_from_img(diff_img, frame)
        fps = int(1 / float((t.perf_counter() - time)))
        fps_acc += 1
        fps_avg = int(fps_acc / (t.perf_counter() - start_t))
        time = t.perf_counter()
    finished = True


threading.Thread(target=lambda a: test(a), args=(["Test"])).start()


def change_debug_mode():
    global debug, fps_avg, fps_acc, time
    debug = not debug
    fps_acc = fps_avg = fps
    time = t.perf_counter()

def reset_first_img_func():
    global reset_first_img
    reset_first_img = True


def submit():
    global morph_size, morph_size_var, threshold_area_var, threshold_area, threshold_area_max_var, threshold_area_max
    morph_size = int(morph_size_var.get())
    threshold_area = int(threshold_area_var.get())
    threshold_area_max = int(threshold_area_max_var.get())

def create_entry(text, variable, row):
    label = tk.Label(top, text=text)
    entry = tk.Entry(top, textvariable=variable, font=('calibre', 10, 'normal'))
    label.grid(row=row, column=0)
    entry.grid(row=row, column=1)

debug_button = tk.Button(text="DEBUG MODE", command=change_debug_mode)
debug_button.grid(row=0, column=0)
reset_first_img_button = tk.Button(text="RESET DIFF IMAGE", command=reset_first_img_func)
reset_first_img_button.grid(row=0, column=1)
submit_button = tk.Button(text="Submit entries", command=submit)
submit_button.grid(row=0, column=2)

create_entry("Morph size: ", morph_size_var, 1)
create_entry("threshold min", threshold_area_var, 2)
create_entry("threshold_max", threshold_area_max_var, 3)
while not finished:
    cv2.waitKey(1)
    cv.imshow('Window_Show', frame_to_show)
    top.update()
