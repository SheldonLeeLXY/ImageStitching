# -*- coding: utf-8 -*-
# @Time    : 2021/6/29 8:49 PM
# @Author  : Sheldon
# @FileName: ImageStitching.py
# @Software: PyCharm
# @Description: description
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

MIN = 10
startTime = time.time()
img1 = cv2.imread("test1.jpg")
img2 = cv2.imread("test2.jpg")

img1 = cv2.resize(img1, (742, 986))
img2 = cv2.resize(img2, (742, 986))

# surf = cv2.xfeatures2d.SURF_create(10000, nOctaves=4, extended=False, upright=True)
surf=cv2.xfeatures2d.SIFT_create()

kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)

flann = cv2.FlannBasedMatcher(indexParams, searchParams)
match = flann.knnMatch(des1, des2, k=2)

good = []
matchesMask = [[0,0] for i in range (len(match))]
for i, (m,n) in enumerate(match):
    if (m.distance<0.75*n.distance):
        good.append(m)
        matchesMask[i] = [1, 0]

drawParams = dict(matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=0)
matchImage = cv2.drawMatchesKnn(img1, kp1, img2, kp2, match, None, **drawParams)

if len(good) > MIN:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ano_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, ano_pts, cv2.RANSAC, 5.0)
    warpImg = cv2.warpPerspective(img2, np.linalg.inv(M), (img1.shape[1] + img2.shape[1], img2.shape[0]))
    direct = warpImg.copy()
    # print(img1.shape)
    # print(direct.shape)
    direct[0:img1.shape[0], 0:img1.shape[1]] = img1
    simple = time.time()

    rows, cols = img1.shape[:2]

    for col in range(0, cols):
        if img1[:, col].any() and warpImg[:, col].any():  # 开始重叠的最左端
            left = col
            break
    for col in range(cols - 1, 0, -1):
        if img1[:, col].any() and warpImg[:, col].any():  # 重叠的最右一列
            right = col
            break

    res = np.zeros([rows, cols, 3], np.uint8)
    for row in range(0, rows):
        for col in range(0, cols):
            if not img1[row, col].any():  # 如果没有原图，用旋转的填充
                res[row, col] = warpImg[row, col]
            elif not warpImg[row, col].any():
                res[row, col] = img1[row, col]
            else:
                srcImgLen = float(abs(col - left))
                testImgLen = float(abs(col - right))
                alpha = srcImgLen / (srcImgLen + testImgLen)
                res[row, col] = np.clip(img1[row, col] * (1 - alpha) + warpImg[row, col] * alpha, 0, 255)

    warpImg[0:img1.shape[0], 0:img1.shape[1]] = res
    final = time.time()
    plt.title("Matches")
    plt.imshow(matchImage)
    plt.show()
    img3 = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
    plt.title("Simple Panorma")
    plt.imshow(img3, ), plt.show()
    img4 = cv2.cvtColor(warpImg, cv2.COLOR_BGR2RGB)
    plt.title("Best Panorma")
    plt.imshow(img4, ), plt.show()
    print("simple stich cost %f" % (simple - startTime))
    print("\ntotal cost %f" % (final - startTime))
    # cv2.imwrite("simplepanorma.png", direct)
    # cv2.imwrite("bestpanorma.png", warpImg)

else:
    plt.title("Matches")
    plt.imshow(matchImage)
    plt.show()
    print("not enough matches!")
