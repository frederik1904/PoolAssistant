import cv2
import numpy as np


def find_image_targets(img, target, threshold=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, target, cv2.TM_CCOEFF_NORMED)
    return np.where(res >= threshold)


def squares_on_img(img, targets, w, h):
    for pt in zip(*targets[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)


filename = 'test_images\\Pool_table_from_above.png'
template_name = 'templates\\tt';
img = cv2.imread(filename)
src_points = []
dst_points = np.float32([(0,0), (512,0), (512,512), (0,512)])
for i in range(1, 5):
    template = cv2.imread(template_name + str(i) + '.png', 0)
    w, h = template.shape[::-1]
    targets = find_image_targets(img, template, 0.7)
    for pt in targets:
        src_points.append(pt)
    squares_on_img(img, targets, w, h)

M, mask = cv2.findHomography(np.float32(src_points), dst_points, cv2.RANSAC, 5.0)
matchesMask = mask.ravel.tolist()

dst = cv2.perspectiveTransform()
cv2.imwrite('output\\res.png', img)
