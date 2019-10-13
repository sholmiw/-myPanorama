import cv2
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
#import datetime
import math
import os
from collections import OrderedDict
def readImg (path,flag):
    images = []
    filename = ""
    Dict = OrderedDict()
    for fileName in os.listdir(path):
        if filename is None:
            filename = fileName[:-5]
        img = cv2.imread(os.path.join(path, fileName), flag)
        if img is not None:
            images.append(img)
    Dict.setdefault(filename, []).extend(images)
    return Dict
def ssd(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) / len(a)
def findFeatures(img):
    print("Finding Features...")
    #sift = cv2.SIFT()
    #sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with ORB
    #keypoints, descriptors = sift.detectAndCompute(img, None)
    # cv2.imwrite('sift_keypoints.png', img)
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    keypoints, descriptors = orb.detectAndCompute(img, None)
    #img = cv2.drawKeypoints(img, keypoints, None)
    return keypoints, descriptors
def applyHomographyP(xy, H):
    # per point
    xyz = []
    for a in xy:
        xyz.append(a)
    xyz.append(1)
    # convert to a column vector
    xyz_transpose = np.transpose(np.asarray(xyz, dtype=np.float32))
    # Apply homography matrix
    new_xyz = np.matmul(H, xyz_transpose)
    # Homogeneous to Cartesian conversion
    z = new_xyz[2]
    x = int(new_xyz[0] / z)
    y = int(new_xyz[1] / z)
    return [x, y]
def matchFeatures2(kp1, kp2, desc1, desc2):
    #matchFeatures using cv2.BFMatcher
    pos1 = []
    pos2 = []
    matcher = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = matcher.match(desc1, desc2)
    # -- Filter matches using the Lowe's ratio test
    min_dist = 200
    for match in matches:
        if match.distance < min_dist:  # more accurate is doing:  match.distance <  2* min_dist
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            [x1, y1] = kp1[img1_idx].pt
            [x2, y2] = kp2[img2_idx].pt
            pos1.append([x1, y1])  # save pos1
            pos2.append([x2, y2])  # save pos2
    return pos1, pos2
def matchFeatures(kp1, kp2, desc1, desc2):
    matches = fMatcher(desc1, desc2)
    Minmum_Matchs = 4 #for 8 perms
    if len(matches) > Minmum_Matchs:
        coordList = []
        pos1 = []
        pos2 = []
        for m in matches:
            (x1, y1) = kp1[m[1]].pt
            pos1.append([x1, y1])
            (x2, y2) = kp2[m[2]].pt
            pos2.append([x2, y2])
            coordList.append([x1, y1, x2, y2])
        return pos1, pos2
    else:
        print("Not enough matches are found, only:" ,len(matches), "found")
def fMatcher(desc1, desc2):
    matches =[]
    idx_1 = 0
    for i in desc1:
        allMatches = []
        idx_2 = 0
        for j in desc2:
            d = cv2.norm(i, j) #distance in absolute value. give us better result's when our ssd aproch
            key = [d, idx_1, idx_2] #distance , index in desc1 , index in desc2
            allMatches.append(key)
            idx_2 += 1
        idx_1 += 1
        allMatches.sort()
        #print(allMatches[0:2])
        matches.append(allMatches[0:2]) #take closest matche
    goodMatche = []
    for m, n in matches:
        if m[0] < 0.8 * n[0]: #play with the number to 0.8 leter 0.5 not  alweys work
            goodMatche.append(m)
    return goodMatche

    """
    for i in range(0, kp1size):
        sMin = 1000000
        sMin_index = 0
        for j in range(0, kp2size):
            sCurr = ssd(desc1[i], desc2[j])
            if sCurr < sMin:
                sMin = sCurr
                sMin_index = j

        pos1.append(kp1[i].pt)
        pos2.append(kp2[sMin_index].pt)
    return pos1, pos2
  """
def calculateHomography(cor):
    # loop through correspondences and create assemble matrix
    matrixA = [[cor[0][0][0], cor[0][0][1], 1, 0, 0, 0, -cor[0][1][0] * cor[0][0][0], -cor[0][1][0] * cor[0][0][1],
                -cor[0][1][0]],
               [0, 0, 0, cor[0][0][0], cor[0][0][1], 1, -cor[0][1][1] * cor[0][0][0], -cor[0][1][1] * cor[0][0][1],
                -cor[0][1][1]],
               [cor[1][0][0], cor[1][0][1], 1, 0, 0, 0, -cor[1][1][0] * cor[1][0][0], -cor[1][1][0] * cor[1][0][1],
                -cor[1][1][0]],
               [0, 0, 0, cor[1][0][0], cor[1][0][1], 1, -cor[1][1][1] * cor[1][0][0], -cor[1][1][1] * cor[1][0][1],
                -cor[1][1][1]],
               [cor[2][0][0], cor[2][0][1], 1, 0, 0, 0, -cor[2][1][0] * cor[2][0][0], -cor[2][1][0] * cor[2][0][1],
                -cor[2][1][0]],
               [0, 0, 0, cor[2][0][0], cor[2][0][1], 1, -cor[2][1][1] * cor[2][0][0], -cor[2][1][1] * cor[2][0][1],
                -cor[2][1][1]],
               [cor[3][0][0], cor[3][0][1], 1, 0, 0, 0, -cor[3][1][0] * cor[3][0][0], -cor[3][1][0] * cor[3][0][1],
                -cor[3][1][0]],
               [0, 0, 0, cor[3][0][0], cor[3][0][1], 1, -cor[3][1][1] * cor[3][0][0], -cor[3][1][1] * cor[3][0][1],
                -cor[3][1][1]]]
    # svd composition
    u, s, v = np.linalg.svd(matrixA)
    # reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))
    # normalize and now we have h
    h = (1 / h.item(8)) * h
    return h
def ransacHomography(pos1, pos2, numIters, inlierTol):
    inliers = []
    H12 = None
    numRows = len(pos1)
    for i in range(numIters):
        correspondenceList = []
        # find 4 random points to calculate a homography
        rnd_indx = np.random.choice(numRows, 4, replace=False)
        for r in rnd_indx:
            correspondenceList.append([pos1[r], pos2[r]])
        # call the homography function on those points
        h = calculateHomography(correspondenceList)
        inliers_list = []

        for j in range(len(pos2)):
            esP2 = applyHomographyP(pos1[j], h)  # pos1 to pos2 transform

            d = ((((pos2[j][0] - esP2[0]) ** 2) + ((pos2[j][1] - esP2[1]) ** 2)) ** 0.5)

            if d < inlierTol:
                inliers_list.append(j)

        if len(inliers_list) > len(inliers):
            inliers = inliers_list
            H12 = h
    return H12, inliers
def displayMatches(img1,img2,pos1,pos2,inlind):
    # Create a new output image that concatenates the two images together
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    #we sort inlind for quick run
    inlind = sorted(inlind)
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
    # For each pair of points we have between both images
    # connect a line between them
    lin_index=0
    inlindSize = len(inlind)
    for i in range (len(pos1)):
        if lin_index==inlindSize:
            break
        if i == inlind[lin_index]:
            # Draw yellow  line
            cv2.line(out, (int(pos1[i][0]),int(pos1[i][1])), (int(pos2[i][0])+cols1,int(pos2[i][1])), (0, 0, 255), 1)
            lin_index+=1
        else:
        # Draw blue  line
            cv2.line(out, (int(pos1[i][0]),int(pos1[i][1])), (int(pos2[i][0])+cols1,int(pos2[i][1])), (0, 255, 255), 1)

    cv2.imshow("out", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out
def trim(frame):
    # trims the black residue in the panorama image from
    #https://pylessons.com/OpenCV-image-stiching-continue/
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
        #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame
def accumulateHomographies(Hpair, m):
    HpairLen=len(Hpair)+1
    htot = [np.ones((3, 3))] * (HpairLen)
    for i in range(HpairLen):
        if i == m:
            htot[i] = np.eye(3)
        elif i < m:
            for ind in range(i, m):
                htot[i] = htot[i].dot(Hpair[ind])
        elif i > m:
            for ind in range(m, i):
                htot[i] = htot[i].dot(np.linalg.inv(Hpair[ind]))
    return htot
def renderPanorama( imList , Hs):
    warpedImages = []
    for i in range(len(imList)):
        img = imList[i]
        warped = cv2.warpPerspective(img, Hs[i], (img.shape[1], img.shape[0]))
        warped[0:img.shape[0],0:img.shape[1]] = img
        warpedImages.append(warped)
    stitcherO = cv2.Stitcher_create()
    (state, panorama) = stitcherO.stitch(warpedImages)
    if state != 0: # ==0 -> didnt work
        panorama = None
    return trim(panorama)

#tests:
def generatePanorama1():
    """ not working with cv2.Stitcher
    img1c = cv2.cvtColor(cv2.imread('right.jpeg',3)  , cv2.COLOR_BGR2RGB) / 255
    img1 = cv2.cvtColor(cv2.imread('right.jpeg',0) , cv2.COLOR_BGR2GRAY)
    """
    #test 1
    img1c = cv2.imread('backyard3.jpg',3)
    img1 = cv2.imread('backyard3.jpg', 0)
    img2c = cv2.imread('backyard2.jpg', 3)
    img2 = cv2.imread('backyard2.jpg', 0)
    img3c = cv2.imread('backyard1.jpg', 3)
    img3 = cv2.imread('backyard1.jpg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    kp3, desc3 = findFeatures(img3)

    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h,maxInlier = ransacHomography(pos1, pos2,1000,1)

    ##parspctive use
    #dst = cv2.warpPerspective(img1c, h, (img2c.shape[1] + img1c.shape[1], img2c.shape[0]))  # img2 left to img1
    #  #dst = cv2.warpPerspective(img1,H,(img2.shape[0],img2.shape[1] + img1.shape[1]))
    #plt.subplot(122), plt.imshow(dst), plt.title("Warped Image")
    #plt.show()
    #plt.figure()
    #dst[0:img2c.shape[0], 0:img2c.shape[1]] = img2c  # add bluring
    #dst=trim(dst)
    #cv2.imwrite("output.jpg", dst)
    #plt.imshow(dst)
    #plt.show()
    #print(h)
    rH=[]
    rH.append(h)
    #FOR 3 PIC:
    pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
    rH.append(h)

    out = displayMatches(img1, img2, pos1, pos2, maxInlier)
    imList =[img1c,img2c,img3c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2)- 1)
    #print(Hs)
    out = renderPanorama(imList, Hs)
    im_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpg", im_rgb)
    plt.imshow(im_rgb)
    plt.show()

#generatePanorama1()

def generatePanorama2():
    # test2:
    img1c = cv2.imread('neilR.jpg', 3)
    img1 = cv2.imread('neilR.jpg', 0)
    img2c = cv2.imread('neilL.jpg', 3)
    img2 = cv2.imread('neilL.jpg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)

    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)

    rH = []
    rH.append(h)
    out = displayMatches(img1, img2, pos1, pos2, maxInlier)
    imList = [img1c, img2c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2) - 1)
    # print(Hs)
    out = renderPanorama(imList, Hs)
    im_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpg", im_rgb)
    plt.imshow(im_rgb)
    plt.show()

#generatePanorama2()

def generatePanorama3():
    # test3:
    img1c = cv2.imread('right.jpeg', 3)
    img1 = cv2.imread('right.jpeg', 0)
    img2c = cv2.imread('left.jpeg', 3)
    img2 = cv2.imread('left.jpeg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)

    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)

    rH = []
    rH.append(h)
    out = displayMatches(img1, img2, pos1, pos2, maxInlier)
    imList = [img1c, img2c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2) - 1)
    # print(Hs)
    out = renderPanorama(imList, Hs)
    im_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpg", im_rgb)
    plt.imshow(im_rgb)
    plt.show()

#generatePanorama3()

def generatePanorama4():
    #test 4
    img1c = cv2.imread('panL.jpg',3)
    img1 = cv2.imread('panL.jpg', 0)
    img2c = cv2.imread('panM.jpg', 3)
    img2 = cv2.imread('panM.jpg', 0)
    img3c = cv2.imread('panR.jpg', 3)
    img3 = cv2.imread('panR.jpg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    kp3, desc3 = findFeatures(img3)

    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h,maxInlier = ransacHomography(pos1, pos2,1000,1)

    rH=[]
    rH.append(h)
    #FOR 3 PIC:

    pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
    rH.append(h)

    #out = displayMatches(img1, img2, pos1, pos2, maxInlier)
    imList =[img1c,img2c,img3c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2)- 1)
    #print(Hs)
    out = renderPanorama(imList, Hs)
    cv2.imwrite("output.jpg", out)
    plt.imshow(out)
    plt.show()

#generatePanorama4()

def generatePanorama5():

    img1c = cv2.imread('lef.jpeg',3)
    img1 = cv2.imread('lef.jpeg', 0)
    img2c = cv2.imread('mid.jpeg', 3)
    img2 = cv2.imread('mid.jpeg', 0)
    img3c = cv2.imread('ret.jpeg', 3)
    img3 = cv2.imread('ret.jpeg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    kp3, desc3 = findFeatures(img3)

    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h,maxInlier = ransacHomography(pos1, pos2,1000,1)

    rH=[]
    rH.append(h)
    #FOR 3 PIC:
    pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
    rH.append(h)

    out = displayMatches(img1, img2, pos1, pos2, maxInlier)# here it's to big
    imList =[img1c,img2c,img3c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2)- 1)
    #print(Hs)
    out = renderPanorama(imList, Hs)
    im_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpg", im_rgb)
    plt.imshow(im_rgb)
    plt.show()

#generatePanorama5()

def generatePanorama6():

    img1c = cv2.imread('office1.jpg',3)
    img1 = cv2.imread('office1.jpg', 0)
    img2c = cv2.imread('office2.jpg', 3)
    img2 = cv2.imread('office2.jpg', 0)
    img3c = cv2.imread('office3.jpg', 3)
    img3 = cv2.imread('office3.jpg', 0)
    img4c = cv2.imread('office4.jpg', 3)
    img4 = cv2.imread('office4.jpg', 0)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    kp3, desc3 = findFeatures(img3)
    kp4, desc4 = findFeatures(img4)
    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h,maxInlier = ransacHomography(pos1, pos2,1000,1)

    rH=[]
    rH.append(h)
    #FOR 4 PIC:
    pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
    rH.append(h)
    pos1, pos2 = matchFeatures(kp3, kp4, desc3, desc4)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
    rH.append(h)

    #out = displayMatches(img1, img2, pos1, pos2, maxInlier)# here it's to big
    imList =[img1c,img2c,img3c]
    Hs = accumulateHomographies(rH, math.ceil(len(imList) / 2)- 1)
    #print(Hs)
    out = renderPanorama(imList, Hs)
    im_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imwrite("output.jpg", im_rgb)
    plt.imshow(im_rgb)
    plt.show()

#generatePanorama6()

def generatePanorama7():
    path = "C:\im"
    imaGbr = readImg(path, cv2.IMREAD_COLOR)
    imGray = readImg(path, cv2.IMREAD_GRAYSCALE)

    for imgKey in imGray:
        counter = 1
        imagesGBR = imaGbr[imgKey]
        imagesGray = imGray[imgKey]
        LenimagesGray = len(imagesGray)

    for k in range(LenimagesGray - 1):
        fileName = imgKey + str(counter)

        kp1, desc1 = findFeatures(imagesGray[k])
        kp2, desc2 = findFeatures(imagesGray[k + 1])
        # kp3, desc3 = findFeatures(img3)

        pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
        rH = []
        rH.append(h)
        # FOR 3 PIC:
        # pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
        rH.append(h)

        # out = displayMatches(img1, img2, pos1, pos2, maxInlier)
        counter += 1
    # imList = [img1c, img2c, img3c]
    Hs = accumulateHomographies(rH, math.ceil(LenimagesGray / 2) - 1)
    # print(Hs)
    out = renderPanorama(imagesGBR, Hs)
    cv2.imwrite("output.jpg", out)
    plt.imshow(out)
    plt.show()

#generatePanorama7()

def generatePanorama(path):
    #path = "C:\im"
    imaGbr = readImg(path, cv2.IMREAD_COLOR)
    imGray = readImg(path, cv2.IMREAD_GRAYSCALE)

    for imgKey in imGray:
        counter = 1
        imagesGBR = imaGbr[imgKey]
        imagesGray = imGray[imgKey]
        LenimagesGray = len(imagesGray)

    for k in range(LenimagesGray - 1):
        fileName = imgKey + str(counter)

        kp1, desc1 = findFeatures(imagesGray[k])
        kp2, desc2 = findFeatures(imagesGray[k + 1])
        # kp3, desc3 = findFeatures(img3)

        pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
        rH = []
        rH.append(h)
        # FOR 3 PIC:
        # pos1, pos2 = matchFeatures(kp2, kp3, desc2, desc3)
        pos1 = np.asarray(pos1)
        pos2 = np.asarray(pos2)
        h, maxInlier = ransacHomography(pos1, pos2, 1000, 1)
        rH.append(h)

        # out = displayMatches(img1, img2, pos1, pos2, maxInlier)
        counter += 1
    # imList = [img1c, img2c, img3c]
    Hs = accumulateHomographies(rH, math.ceil(LenimagesGray / 2) - 1)
    # print(Hs)
    out = renderPanorama(imagesGBR, Hs)
    cv2.imwrite("output.jpg", out)
    plt.imshow(out)
    plt.show()

if _name_ == '_main_':
    generatePanorama('C:\im') #give folder with pic to work on
    # generatePanorama1()
    # generatePanorama2()
    # generatePanorama3()
    # generatePanorama4()
    # generatePanorama5()
    # generatePanorama6()
    # generatePanorama7() #with givan path