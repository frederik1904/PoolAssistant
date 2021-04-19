import cv2
import numpy as np


def find_image_targets(img, target, threshold=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF_NORMED)
    return np.where(res >= threshold)


def squares_on_img(img, targets, w, h):
    for pt in zip(*targets[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


def find_diff(img1, img2):
    diff = cv2.subtract(img1, img2)
    conv_hsv_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(conv_hsv_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    diff[mask != 255] = [255, 255, 255]
    return diff


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    img2, contours, hireacy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite('output/res3.png', img2)


filename = 'test_images/Pool_table_from_above.png'
filename2 = 'test_images/Pool_table_from_above_clean.png'
template_name = 'templates/tt'
img = cv2.imread(filename)
img2 = cv2.imread(filename2)
src_points = []
width, height = (1024, 512)
dst_points = np.float32([(0,0), (width,0), (width,height), (0,height)]).reshape(-1,1,2)
for i in range(1, 5):
    template = cv2.imread(template_name + str(i) + '.png', 0)
    w, h = template.shape[::-1]
    targets = find_image_targets(img, template, 0.65)
    src_points.append(list(zip(*targets[::-1]))[0])
    squares_on_img(img, targets, w, h)
src_points.append(src_points.pop(0))
src_points.append(src_points.pop(0))
src_points.append(src_points.pop(0))
M, mask = cv2.findHomography(np.float32(src_points).reshape(-1,1,2), dst_points, cv2.RANSAC,5.0)
im_dst = cv2.warpPerspective(img, M, (width, height))

diff_img = find_diff(im_dst, img2)
find_contours(diff_img)
cv2.imwrite('output/res3.png', diff_img)
cv2.imwrite("output/res2.png", im_dst)
cv2.imwrite('output/res.png', img)
