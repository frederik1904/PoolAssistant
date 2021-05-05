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
debug = False
fullscreen = False
reset_first_img = False
frame_to_show = 0
debug_frame = 0
cue_frame = 0
finished = False
morph_size = 5
morph_size_var = tk.StringVar()
morph_size_var.set(str(morph_size))
testName = 0
threshold_area = 500
threshold_area_max = 5000
threshold_area_var = tk.StringVar()
threshold_area_max_var = tk.StringVar()
threshold_area_var.set(str(threshold_area))
threshold_area_max_var.set(str(threshold_area_max))
width, height = 1024, 512
offset_x_h = -30
offset_x_h_var = tk.StringVar()
offset_x_h_var.set(str(offset_x_h))
offset_x_l = 25
offset_x_l_var = tk.StringVar()
offset_x_l_var.set(str(offset_x_l))
offset_y_h = 2
offset_y_h_var = tk.StringVar()
offset_y_h_var.set(str(offset_y_h))
offset_y_l = -8
offset_y_l_var = tk.StringVar()
offset_y_l_var.set(str(offset_y_l))

threshold = 80
threshold_var = tk.StringVar()
threshold_var.set(str(threshold))

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParams = cv2.aruco.DetectorParameters_create()


def get_aruco(marker_id, size):
    new_image = np.zeros((size, size, 1), np.uint8)
    cv2.aruco.drawMarker(arucoDict, marker_id, size, new_image)
    new_image = cv2.cvtColor(new_image, cv.COLOR_GRAY2RGB)
    return new_image


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
    return diff


def find_boxes_from_img(img, output, w, h, last_out):
    global debug_frame, debug
    ret, thresh_gray = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), threshold, 255, cv.THRESH_BINARY)
    tg_copy = thresh_gray.copy()
    # thresh_gray = thresh_gray - cv.cvtColor(last_out, cv.COLOR_BGR2GRAY)
    contours, hier = cv.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    black_canvas = np.zeros((h, w, 3), np.uint8)
    for cont in contours:
        area = cv.contourArea(cont)

        if area < threshold_area:
            cv.fillPoly(thresh_gray, pts=[cont], color=0)
            continue

        rect = cv.minAreaRect(cont)
        (x, y), (w, h), angle = rect
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 2.5:  # or w > 100 or h > 100:
            cv2.fillPoly(thresh_gray, pts=[cont], color=0)
            continue

    thresh_gray2 = cv2.morphologyEx(thresh_gray, cv.MORPH_CLOSE,
                                    cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size)))
    contours, hier = cv.findContours(thresh_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour_cords = []
    for cont in contours:
        area = cv.contourArea(cont)
        if area < threshold_area or area > threshold_area_max:
            cv2.fillPoly(thresh_gray2, pts=[cont], color=0)
            continue
        rect = cv.minAreaRect(cont)
        (x, y), (w, h), _ = rect
        black_canvas = cv.circle(black_canvas, (int(x), int(y)), 25, (255, 255, 255), 5)
        contour_cords.append(cont)
    pos, radius = find_avg_color(output, thresh_gray2)
    find_cue(output, tg_copy, pos)
    black_canvas = cv.circle(black_canvas, pos, radius + 60, (255, 255, 255), 5)

    output = cv.putText(output, "FPS: " + str(fps) + ", AVG: " + str(fps_avg), (0, 500), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255))
    backtorgb1 = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2RGB)
    backtorgb = cv2.cvtColor(thresh_gray2, cv2.COLOR_GRAY2RGB)
    debug_frame = np.vstack([np.hstack([img, backtorgb1]), np.hstack([backtorgb, output])])
    if debug:
        return np.vstack([np.hstack([img, backtorgb1]), np.hstack([backtorgb, output])])
    else:
        return black_canvas


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


def find_avg_color(frame, diff):
    radius = 25
    diff = cv.bitwise_and(frame, frame, mask=diff)
    gray = cv.cvtColor(diff, cv.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    return maxLoc, radius


def find_cue(img, diff, pos):
    global cue_frame
    h, w = 200, 200
    x, y = pos
    dst_points = np.float32([(0, 0), (0, w), (h, w), (h, 0)]).reshape(-1, 1, 2)
    src_points = np.float32([(x - h // 2, y - w // 2), (x - h // 2, y + w // 2), (x + h // 2, y + w // 2),
                             (x + h // 2, y - h // 2)]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_points, dst_points)

    contours, _ = cv.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        if cv2.contourArea(cont) < 10:
            cv2.fillPoly(diff,pts=[cont],color=0)
            continue
        (x, y), (we, he), angle = cv.minAreaRect(cont)
        if min(we, he) == 0:
            continue
        aspect_ratio = max(we, he) / min(we, he)
        if aspect_ratio < 2:  # or w > 100 or h > 100:
            cv2.fillPoly(diff, pts=[cont], color=0)
        else: #This is the cue?


    cue_frame = np.vstack(
        [cv2.warpPerspective(img, M, (h, w)), cv2.cvtColor(cv2.warpPerspective(diff, M, (h, w)), cv2.COLOR_GRAY2RGB)])


def test(a):
    # Load calib values
    global fps, fps_acc, fps_avg, time, frame_to_show, reset_first_img, finished, pik, debug, width, height

    src_points = []
    width, height = (1280, 740)

    cap = cv2.VideoCapture(2)
    src_points = calibrate_board_corners(cap)
    print("Found corners")
    dst_points = np.float32([(0, 0), (width, 0), (width, height), (0, height)]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(np.float32(src_points).reshape(-1, 1, 2), dst_points, cv2.RANSAC, 5.0)
    src_points, dst_points = calibrate_projector(cap, M)
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    reset_first_img = True
    ret, frame = cap.read()
    img2 = frame
    frame_to_show = frame
    img2 = cv2.warpPerspective(img2, M, (width, height))
    lastOutput = black_canvas = np.zeros((height, width, 3), np.uint8)
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.warpPerspective(frame, M, (width, height))
        if reset_first_img:
            frame_to_show = np.zeros((height, width, 3), np.uint8)
            t.sleep(1)
            ret, frame = cap.read()
            frame = cv2.warpPerspective(frame, M, (width, height))
            img2 = frame
            reset_first_img = False
            continue

        diff_img = find_diff(frame, img2)
        frame_to_show = find_boxes_from_img(diff_img, frame, width, height, lastOutput)
        if not debug:
            lastOutput = frame_to_show
        fps = int(1 / float((t.perf_counter() - time)))
        fps_acc += 1
        fps_avg = int(fps_acc / (t.perf_counter() - start_t))
        time = t.perf_counter()
    finished = True


def insert_marker_on_img(marker, img, cord, size):
    img[cord[0]:cord[0] + size, cord[1]:cord[1] + size] = marker
    return img


def calibrate_projector(cap, homography_m):
    global height, width, frame_to_show, debug_frame
    size = 100
    initial_offset = 25
    buffer = 20
    sleep_b = 0.4
    ids_to_find = [2, 3, 4, 5]
    aruco_images = [get_aruco(marker_id, size) for marker_id in ids_to_find]

    cornerDict = {}
    src_points = []
    center = (height // 2, width // 2)
    marker_2 = (center[0] - size - buffer, center[1] - size - buffer)  # Top Left
    marker_3 = (center[0] - size - buffer, center[1] + buffer)  # Top Right
    marker_4 = (center[0] + buffer, center[1] + buffer)  # Bottom Right
    marker_5 = (center[0] + buffer, center[1] - size - buffer)  # Bottom Left
    markers = [
        [marker_2, -initial_offset, -1, False],
        [marker_3, -initial_offset, 1, False],
        [marker_4, initial_offset, -1, False],
        [marker_5, initial_offset, 1, False]
    ]
    img_to_show = np.zeros((height, width, 3), np.uint8)
    img_to_show[:, :] = (255, 255, 255)  # make image white
    for index in range(len(markers)):
        img_to_show = insert_marker_on_img(aruco_images[index], img_to_show, markers[index][0], size)
    frame_to_show = img_to_show
    t.sleep(sleep_b)
    while not all([i[1] == 0 and i[2] == 0 for i in markers]):
        img_to_show = np.zeros((height, width, 3), np.uint8)
        img_to_show[:, :] = (255, 255, 255)  # make image white
        for index in range(len(markers)):
            img_to_show = insert_marker_on_img(aruco_images[index], img_to_show, markers[index][0], size)
        frame_to_show = img_to_show
        t.sleep(sleep_b)
        ret, frame = cap.read()
        # frame = cv.warpPerspective(frame, homography_m, (width, height))
        debug_frame = frame
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

        for marker_id in ids_to_find:
            marker_t = markers[marker_id - 2]
            new_touple = marker_t[0]
            if ids is not None and marker_id in ids:
                if marker_t[1] != 0:  # Checks that x is still being calibrated
                    if marker_t[3]:  # Checks whether we just came from out of bounds
                        marker_t[1] //= 2
                        marker_t[1] *= -1
                        marker_t[3] = False
                    new_touple = (marker_t[1] + marker_t[0][0], marker_t[0][1])
                    if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
                        1] == 0:  # checks the bounds (duh) or Checks if we are about to change to calibrating the y-axis
                        new_touple = insert_buffer(marker_t[0], marker_id, buffer)
                        marker_t[1] = 0
                        marker_t[2] *= initial_offset
                        marker_t[3] = False
                elif marker_t[2] != 0:  # Checks that y is still being calibrated
                    if marker_t[3]:  # Checks whether we just came from out of bounds
                        marker_t[2] //= 2
                        marker_t[2] *= -1
                        marker_t[3] = False
                    new_touple = (marker_t[0][0], marker_t[0][1] + marker_t[2])
                    if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
                        2] == 0:  # Checks if we are about to change to calibrating the y-axis
                        new_touple = insert_buffer(marker_t[0], marker_id, buffer, dir=1)
                        marker_t[2] = 0
                        marker_t[3] = False
            else:  # Marker cant be seen by camera
                if marker_t[1] != 0:  # Checks that x is still being calibrated
                    if not marker_t[3]:  # Checks whether we just came from out of bounds
                        marker_t[1] //= 2
                        marker_t[1] *= -1
                        marker_t[3] = True
                    new_touple = (marker_t[1] + marker_t[0][0], marker_t[0][1])
                    if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
                        1] == 0:  # Checks if we are about to change to calibrating the y-axis or checks the bounds (duh)
                        new_touple = insert_buffer(marker_t[0], marker_id, buffer)
                        marker_t[1] = 0
                        marker_t[2] *= initial_offset
                        marker_t[3] = True
                elif marker_t[2] != 0:  # Checks that y is still being calibrated
                    if not marker_t[3]:  # Checks whether we just came from out of bounds
                        marker_t[2] //= 2
                        marker_t[2] *= -1
                        marker_t[3] = True
                    new_touple = (marker_t[0][0], marker_t[0][1] + marker_t[2])
                    if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
                        2] == 0:  # Checks if we are about to change to calibrating the y-axis
                        new_touple = insert_buffer(marker_t[0], marker_id, buffer, dir=1)
                        marker_t[2] = 0
                        marker_t[3] = True
            marker_t[0] = new_touple

    src_points = []
    dst_points = []
    done = False
    while not done:
        ret, frame = cap.read()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        correct_ids = [mark_id for mark_id in ids if mark_id in ids_to_find]

        if len(correct_ids) < 4:
            print("Didnt find all that i wanted trying again")
            t.sleep(.5)
            continue
        print("Found it!")
        for (corner, marker_id) in zip(corners, ids):
            if marker_id not in ids_to_find:
                continue
            corner = corner.reshape((4, 2))
            marker_id = int(marker_id[0])
            marker_y, marker_x = markers[marker_id - 2][0]
            marker_x = int(marker_x)
            marker_y = int(marker_y)
            src_points.append((corner[marker_id - 2], marker_id))
            if marker_id == 2:
                dst_points.append(((marker_x, marker_y), marker_id))
            elif marker_id == 3:
                dst_points.append(((marker_x + size, marker_y), marker_id))
            elif marker_id == 4:
                touple = (marker_x + size, marker_y + size)
                dst_points.append((touple, marker_id))
            elif marker_id == 5:
                dst_points.append(((marker_x, marker_y + size), marker_id))

        done = True

    src_points.sort(key=lambda x: x[1])
    dst_points.sort(key=lambda x: x[1])
    src_points = [item[0] for item in src_points]
    dst_points = [item[0] for item in dst_points]

    # src_points.append(src_points.pop(1))
    # src_points.append(src_points.pop(1))
    # src_points.append(src_points.pop(1))

    # dst_points.append(dst_points.pop(2))
    # dst_points.append(dst_points.pop(1))

    return np.float32(src_points).reshape(-1, 1, 2), np.float32(dst_points).reshape(-1, 1, 2)


def insert_buffer(touple, id, buffer, dir=0):
    x = y = 0
    if dir == 0:
        x = 1
    else:
        y = 1
    if id == 2:
        return touple[0] + (buffer * x), touple[1] + (buffer * y)
    elif id == 3:
        return touple[0] + (buffer * x), touple[1] - (buffer * y)
    elif id == 4:
        return touple[0] - (buffer * x), touple[1] - (buffer * y)
    elif id == 5:
        return touple[0] - (buffer * x), touple[1] + (buffer * y)
    exit(1337)


# True if it is in bounds false otherwise
def check_bounds(x, y, size, width, height):
    if x < 0 or x + size > width:
        return False
    if y < 0 or y + size > height:
        return False
    return True


def calibrate_board_corners(cap):
    cornerDict = {}
    src_points = []
    while len(src_points) < 4:
        ret, frame = cap.read()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        for (markerCorner, id) in zip(corners, ids):
            if str(id) not in cornerDict:
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                c = topLeft
                if id[0] == 1:
                    c = bottomLeft
                elif id[0] == 69:
                    c = topLeft
                elif id[0] == 420:
                    c = topRight
                elif id[0] == 666:
                    c = topLeft
                cornerDict[str(id)] = (c, id)
                src_points.append([c, id])
    src_points.sort(key=lambda x: x[1])
    src_points = [item[0] for item in src_points]
    return src_points


threading.Thread(target=lambda a: test(a), args=(["Test"])).start()


def change_debug_mode():
    global debug, fps_avg, fps_acc, time
    debug = not debug
    fps_acc = fps_avg = fps
    time = t.perf_counter()


def reset_first_img_func():
    global reset_first_img
    reset_first_img = True


def fullscreen_func():
    global fullscreen
    fullscreen = not fullscreen


def submit():
    global morph_size, morph_size_var, threshold_area_var, threshold_area, threshold_area_max_var, threshold_area_max, pik
    global offset_y_l, offset_x_h, offset_y_h, offset_x_l, offset_y_l_var, offset_x_h_var, offset_x_l_var, offset_y_h_var
    global threshold
    morph_size = int(morph_size_var.get())
    threshold_area = int(threshold_area_var.get())
    threshold_area_max = int(threshold_area_max_var.get())
    offset_x_l = int(offset_x_l_var.get())
    offset_x_h = int(offset_x_h_var.get())
    offset_y_h = int(offset_y_h_var.get())
    offset_y_l = int(offset_y_l_var.get())
    threshold = int(threshold_var.get())


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
fullscreen_button = tk.Button(text="FULSCREEN", command=fullscreen)
fullscreen_button.grid(row=0, column=3)

create_entry("Morph size: ", morph_size_var, 1)
create_entry("threshold min", threshold_area_var, 2)
create_entry("threshold_max", threshold_area_max_var, 3)
create_entry("Min x-offset", offset_x_l_var, 4)
create_entry("Max x-offset", offset_x_h_var, 5)
create_entry("Min y-offset", offset_y_l_var, 6)
create_entry("Max y-offset", offset_y_h_var, 7)
create_entry("Threshold", threshold_var, 8)

cv2.namedWindow("Window_Show")
cv2.moveWindow("Window_Show", 2048, 0)
cv2.namedWindow("Debug_Window_Show")
cv2.moveWindow("Debug_Window_Show", 0, 0)

while not finished:
    cv2.waitKey(1)
    cv.imshow('Window_Show', frame_to_show)
    cv.imshow("Debug_Window_Show", debug_frame)
    cv.imshow("Cue_window_show", cue_frame)

    top.update()
