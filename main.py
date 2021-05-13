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
start_t = t.time()
time = t.time()
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
width, height = (1280, 740)
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

producer_consumer_img = 0
producer_consumer_img_count = 0
producer_consumer_img_lock = threading.Lock()

threshold = 60
threshold_var = tk.StringVar()
threshold_var.set(str(threshold))

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_1000)
arucoParams = cv2.aruco.DetectorParameters_create()


def draw_magic_lines_of_helpiness():
    global outer_M, outer_M_inv, M, debug_frame, img2_outer
    # Step 1: Homography
    debug_frame = outer_wrapped
    #
    # Step 2: Find line
    # Step 3: ????
    # Step 4: Profit
    pass


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
    start_time = t.time()
    ret, thresh_gray = cv.threshold(cv.cvtColor(img, cv.COLOR_BGR2GRAY), threshold, 255, cv.THRESH_BINARY)
    tg_copy = thresh_gray.copy()
    # thresh_gray = thresh_gray - cv.cvtColor(last_out, cv.COLOR_BGR2GRAY)
    contours, hier = cv.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    black_canvas = np.zeros((h, w, 3), np.uint8)
    black_canvas[:, :] = (255, 255, 255)  # not so black now
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
    time_p1 = t.time()
    thresh_gray2 = cv2.morphologyEx(thresh_gray, cv.MORPH_CLOSE,
                                    cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_size, morph_size)))
    contours, hier = cv.findContours(thresh_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contour_cords = []
    for cont in contours:
        area = cv.contourArea(cont)
        if area < threshold_area or area > threshold_area_max:
            cv2.fillPoly(thresh_gray2, pts=[cont], color=0)
            continue
        # rect = cv.minAreaRect(cont)
        # (x, y), (w, h), _ = rect
        # black_canvas = cv.circle(black_canvas, (int(x), int(y)), 25, (255, 255, 255), 5)
        # contour_cords.append(cont)
    time_p2 = t.time()
    pos, radius = find_avg_color(output, thresh_gray2)
    # draw_magic_lines_of_helpiness()
    find_cue(output, tg_copy, pos, black_canvas, radius + 60)
    time_p3 = t.time()
    black_canvas = cv.circle(black_canvas, pos, radius + 60, (0, 255, 255), 5)
    # black_canvas = cv.putText(black_canvas, "FPS: " + str(fps) + ", AVG: " + str(fps_avg), (0, 500), cv2.FONT_HERSHEY_SIMPLEX,
    # 3, (255, 255, 255))
    # output = cv.putText(output, "FPS: " + str(fps) + ", AVG: " + str(fps_avg), (0, 500), cv2.FONT_HERSHEY_SIMPLEX,
    #                    3, (255, 255, 255))
    # backtorgb1 = cv2.cvtColor(thresh_gray, cv2.COLOR_GRAY2RGB)
    # backtorgb = cv2.cvtColor(thresh_gray2, cv2.COLOR_GRAY2RGB)
    # debug_frame = np.vstack([np.hstack([img, backtorgb1]), np.hstack([backtorgb, output])])
    time_p4 = t.time()
    if time_p4 - start_time > 0.016:
        print(
            f"Time diff p1: {time_p1 - start_time}, p2: {time_p2 - time_p1}, p3: {time_p3 - time_p2}, p4: {time_p4 - time_p3}, total: {time_p4 - start_time}")
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


def find_cue(img, diff, pos, frame_to_show,rasmus_the_radius):
    global cue_frame
    h, w = 200, 200
    x, y = pos
    dst_points = np.float32([(0, 0), (0, w), (h, w), (h, 0)]).reshape(-1, 1, 2)
    src_points = np.float32([(x - h // 2, y - w // 2), (x - h // 2, y + w // 2), (x + h // 2, y + w // 2),
                             (x + h // 2, y - h // 2)]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_points, dst_points)
    M_back = np.linalg.inv(M)
    diff = cv2.warpPerspective(diff, M, (h, w))

    contours, _ = cv.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found_cue = False
    for cont in contours:
        if cv2.contourArea(cont) < 100:
            cv2.fillPoly(diff, pts=[cont], color=0)
            continue
        (x, y), (we, he), angle = cv.minAreaRect(cont)
        if min(we, he) == 0:
            continue
        aspect_ratio = max(we, he) / min(we, he)
        if aspect_ratio < 2:  # or w > 100 or h > 100:
            cv2.fillPoly(diff, pts=[cont], color=0)
        else:  # This is the cue?
            found_cue = True
            try:
                cont_to_fill = cont.copy()
                cont = cv2.perspectiveTransform(np.float32(cont), M_back)

                rows, cols = frame_to_show.shape[:2]
                [vx, vy, x, y] = cv.fitLine(cont, cv.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                cv.line(frame_to_show, (cols - 1, righty), (0, lefty), (0, 255, 255), 8)
                cv.circle(frame_to_show, pos, rasmus_the_radius, (255, 255, 255), thickness=-1)
                #cv2.drawContours(frame_to_show, [cont], -1, (255, 255, 255), thickness=-1)
                print("i cri")
            except cv2.error as e:
                print(e)
                continue
            except OverflowError:
                continue

    #cue_frame = np.vstack(
    #    [cv2.warpPerspective(img, M, (h, w)), cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)])


def img_producer():
    global producer_consumer_img_lock, producer_consumer_img, producer_consumer_img_count, finished
    cap = cv2.VideoCapture(2)
    # cap = cv2.VideoCapture('test_images/tm2.mov')
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv.CAP_PROP_FPS, 60)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    ret, frame = cap.read()
    time_acc = 0
    while not finished and ret:
        start = t.time()

        ret, frame = cap.read()

        time_acc += t.time() - start
        # print(t.time() - start)
        producer_consumer_img_lock.acquire()
        producer_consumer_img = frame
        producer_consumer_img_count += 1
        # print(time_acc / producer_consumer_img_count)
        producer_consumer_img_lock.release()
        # t.sleep(0.033)


def get_image(img_count):
    global producer_consumer_img, producer_consumer_img_count, producer_consumer_img_lock
    img = 0
    img_number = 0
    while True:
        producer_consumer_img_lock.acquire()
        if img_count >= producer_consumer_img_count:
            producer_consumer_img_lock.release()
            continue
        img = producer_consumer_img
        img_number = producer_consumer_img_count
        producer_consumer_img_lock.release()
        break
    return img, img_number


def img_consumer():
    # Load calib values
    global fps, fps_acc, fps_avg, time, frame_to_show, reset_first_img, finished, pik, debug, width, height
    global producer_consumer_img_lock, producer_consumer_img, producer_consumer_img_count
    global outer_M, outer_M_inv
    global M, M_inv
    global img2_outer, outer_wrapped

    outer_src_points = calibrate_board_corners()
    outer_dst_points = np.float32([(0, 0), (width, 0), (width, height), (0, height)]).reshape(-1, 1, 2)
    outer_M, mask = cv2.findHomography(np.float32(outer_src_points).reshape(-1, 1, 2), outer_dst_points, cv2.RANSAC,
                                       5.0)
    outer_M_inv = np.linalg.inv(outer_M)
    src_points, dst_points = calibrate_projector()
    M, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    M_inv = np.linalg.inv(M)
    reset_first_img = True
    frame, frame_count = get_image(0)
    img2 = frame
    frame_to_show = frame
    img2 = cv2.warpPerspective(img2, M, (width, height))
    # img2_outer = cv2.warpPerspective(frame_to_show, outer_M, (1024,512))
    lastOutput = black_canvas = np.zeros((height, width, 3), np.uint8)
    time = t.time()
    while not finished:
        frame, frame_count = get_image(frame_count)
        # outer_wrapped = cv2.warpPerspective(frame, outer_M, (1920, 1080))
        time_s = t.time()
        frame = cv2.warpPerspective(frame, M, (width, height))
        time_p1 = t.time()
        if reset_first_img:
            frame_to_show = np.zeros((height, width, 3), np.uint8)
            frame_to_show[:, :] = (255, 255, 255)  # not so black now
            t.sleep(1)
            frame, frame_count = get_image(frame_count)
            # img2_outer = cv2.warpPerspective(frame, outer_M, (1024, 512))
            frame = cv2.warpPerspective(frame, M, (width, height))
            img2 = frame
            reset_first_img = False
            continue

        diff_img = find_diff(frame, img2)
        frame_to_show = find_boxes_from_img(diff_img, frame, width, height, lastOutput)
        if not debug:
            lastOutput = frame_to_show
        fps = int(1 / float((t.time() - time)))
        fps_acc += 1
        fps_avg = int(fps_acc / (t.time() - start_t))
        time = t.time()

        # print(fps_avg, fps, f"Time spent warping: {time_p1 - time_s}, Time spent on img: {t.time() - time_s}")
    finished = True


def insert_marker_on_img(marker, img, cord, size):
    img[cord[0]:cord[0] + size, cord[1]:cord[1] + size] = marker
    return img


def calibrate_projector():
    global height, width, frame_to_show, debug_frame
    size = 100
    initial_offset = 50
    buffer = 120
    sleep_b = 0.4
    ids_to_find = [2, 3, 4, 5]
    aruco_images = [get_aruco(marker_id, size) for marker_id in ids_to_find]

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
    # while not all([i[1] == 0 and i[2] == 0 for i in markers]):
    #     img_to_show = np.zeros((height, width, 3), np.uint8)
    #     img_to_show[:, :] = (255, 255, 255)  # make image white
    #     for index in range(len(markers)):
    #         img_to_show = insert_marker_on_img(aruco_images[index], img_to_show, markers[index][0], size)
    #     frame_to_show = img_to_show
    #     t.sleep(sleep_b)
    #     ret, frame = cap.read()
    #     # frame = cv.warpPerspective(frame, homography_m, (width, height))
    #     debug_frame = frame
    #     (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    #
    #     for marker_id in ids_to_find:
    #         marker_t = markers[marker_id - 2]
    #         new_touple = marker_t[0]
    #         if ids is not None and marker_id in ids:
    #             if marker_t[1] != 0:  # Checks that x is still being calibrated
    #                 if marker_t[3]:  # Checks whether we just came from out of bounds
    #                     marker_t[1] //= 2
    #                     marker_t[1] *= -1
    #                     marker_t[3] = False
    #                 new_touple = (marker_t[1] + marker_t[0][0], marker_t[0][1])
    #                 if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
    #                     1] == 0:  # checks the bounds (duh) or Checks if we are about to change to calibrating the y-axis
    #                     new_touple = insert_buffer(marker_t[0], marker_id, buffer)
    #                     marker_t[1] = 0
    #                     marker_t[2] *= initial_offset
    #                     marker_t[3] = False
    #             elif marker_t[2] != 0:  # Checks that y is still being calibrated
    #                 if marker_t[3]:  # Checks whether we just came from out of bounds
    #                     marker_t[2] //= 2
    #                     marker_t[2] *= -1
    #                     marker_t[3] = False
    #                 new_touple = (marker_t[0][0], marker_t[0][1] + marker_t[2])
    #                 if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
    #                     2] == 0:  # Checks if we are about to change to calibrating the y-axis
    #                     new_touple = insert_buffer(marker_t[0], marker_id, buffer, dir=1)
    #                     marker_t[2] = 0
    #                     marker_t[3] = False
    #         else:  # Marker cant be seen by camera
    #             if marker_t[1] != 0:  # Checks that x is still being calibrated
    #                 if not marker_t[3]:  # Checks whether we just came from out of bounds
    #                     marker_t[1] //= 2
    #                     marker_t[1] *= -1
    #                     marker_t[3] = True
    #                 new_touple = (marker_t[1] + marker_t[0][0], marker_t[0][1])
    #                 if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
    #                     1] == 0:  # Checks if we are about to change to calibrating the y-axis or checks the bounds (duh)
    #                     new_touple = insert_buffer(marker_t[0], marker_id, buffer)
    #                     marker_t[1] = 0
    #                     marker_t[2] *= initial_offset
    #                     marker_t[3] = True
    #             elif marker_t[2] != 0:  # Checks that y is still being calibrated
    #                 if not marker_t[3]:  # Checks whether we just came from out of bounds
    #                     marker_t[2] //= 2
    #                     marker_t[2] *= -1
    #                     marker_t[3] = True
    #                 new_touple = (marker_t[0][0], marker_t[0][1] + marker_t[2])
    #                 if not check_bounds(new_touple[1], new_touple[0], size, width, height) or marker_t[
    #                     2] == 0:  # Checks if we are about to change to calibrating the y-axis
    #                     new_touple = insert_buffer(marker_t[0], marker_id, buffer, dir=1)
    #                     marker_t[2] = 0
    #                     marker_t[3] = True
    #         marker_t[0] = new_touple

    src_points = []
    dst_points = []
    done = False
    frame_count = 0
    while not done:
        frame, frame_count = get_image(frame_count)
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
        if ids is None:
            t.sleep(.5)
            continue
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


def calibrate_board_corners():
    cornerDict = {}
    src_points = []
    frame_count = 0
    while len(src_points) < 4:
        frame, frame_count = get_image(frame_count)
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


threading.Thread(target=img_producer).start()
threading.Thread(target=img_consumer).start()


def change_debug_mode():
    global debug, fps_avg, fps_acc, time
    debug = not debug
    fps_acc = fps_avg = fps
    time = t.perf_counter()


def reset_first_img_func():
    global reset_first_img
    reset_first_img = True


def fullscreen_func():
    global finished
    finished = True


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
fullscreen_button = tk.Button(text="FULSCREEN", command=fullscreen_func)
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
# cv2.namedWindow("Debug_Window_Show")
# cv2.moveWindow("Debug_Window_Show", 0, 0)

while not finished:
    cv2.waitKey(1)
    cv.imshow('Window_Show', frame_to_show)
    #cv.imshow("Debug_Window_Show", debug_frame)
    #cv.imshow("Cue_window_show", cue_frame)

    top.update()
