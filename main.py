import cv2
import numpy as np


def find_image_targets(img, target, threshold=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF_NORMED)
    return np.where(res >= threshold)


def squares_on_img(img, targets, w, h):
    for pt in zip(*targets[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


filename = 'test_images/Pool_table_from_above.png'
template_name = 'templates/tt';
img = cv2.imread(filename)
src_points = []
dst_points = np.float32([(0,0), (512,0), (512,512), (0,512)]).reshape(-1,1,2)
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
im_dst = cv2.warpPerspective(img, M, (512, 512))
cv2.imwrite("output/res2.png", im_dst)


cv2.imwrite('output/res.png', img)
