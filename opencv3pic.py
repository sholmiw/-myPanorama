import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def panoramaOC(name1, name2):
    img1 = cv2.imread(name1)
    #img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1g = cv2.imread(img1, 0)
    img2 = cv2.imread(name2)
    img2g = cv2.imread(img2, 0)
    #img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1g, None)
    kp2, des2 = sift.detectAndCompute(img2g, None)
    #or

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)  # can change to brute force or alse

    # Apply ratio test - we need ransac
    good = []
    for m in matches:
        if m[0].distance < 0.5 * m[1].distance:
            good.append(m)
    matches = np.asarray(good)
    if len(matches[:, 0]) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1, 1, 2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        # print (H)
    else:
        raise AssertionError("Can’t find enough keypoints.")

    dst = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))  # img2 left to img1
    # dst = cv2.warpPerspective(img1,H,(img2.shape[0],img2.shape[1] + img1.shape[1]))
    plt.subplot(122), plt.imshow(dst), plt.title("Warped Image")
    plt.show()
    plt.figure()
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    cv2.imwrite("output.jpg", dst)
    plt.imshow(dst)
    plt.show()

#we make panorama from to and use the new one for the next.
panoramaOC("\data\inp\examples\backyard3.jpg","\data\inp\examples\backyard2.jpg")
panoramaOC("\data\inp\examples\output.jpg","backyard1.jpg")

