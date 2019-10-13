import cv2
import numpy as np
import getopt
import sys
import random
import matplotlib.pyplot as plt
import glob
import datetime
import math

def ssd(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) / len(a)
def findFeatures(img):
    print("Finding Features...")
    #sift = cv2.SIFT()
    #sift = cv2.xfeatures2d.SIFT_create()
    #keypoints, descriptors = sift.detectAndCompute(img, None)
    # print("kkkkk",keypoints)
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    keypoints, descriptors = orb.detectAndCompute(img, None)
    #img = cv2.drawKeypoints(img, keypoints, None)
    #cv2.imwrite('sift_keypoints.png', img)
    # print("pppp",keypoints[0].pt)
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
            # print(correspondenceList)
        # call the homography function on those points
        h = calculateHomography(correspondenceList)
        inliers_list = []

        for j in range(len(pos2)):
            esP2 = applyHomographyP(pos1[j], h)  # pos1 to pos2 transform
            #error = (pos2[j] - esP2)
            # error = np.linalg.norm(error)
            # print(error)
            d = ((((pos2[j][0] - esP2[0]) ** 2) + ((pos2[j][1] - esP2[1]) ** 2)) ** 0.5)
            # print(d)
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

def accumulateHomographies (Hpair, m):
    HpairLen = len(Hpair) + 1
    Htot = np.array([1, HpairLen])
    Htot[m] = np.eye(3)
    for i in range(m - 1, 1, -1):
        Hi = np.dot(Htot[i + 1], Htot[i])
        Hi = (1 / Hi[3, 3]) * Hi
        Htot[i] = Hi
    for i in range(m, HpairLen - 1):
        Hi = Htot[i] / Hpair[i]
        Hi = (1 / Hi[3, 3]) * Hi
        Htot[i + 1] = Hi
    # print("htot", Htot)
    return Htot

def renderPanorama(im, H):
    imLen = len(im)
    minX = sys.maxsize
    minY = sys.maxsize
    maxX = 0
    maxY = 0
    for i in range(imLen):
        edges = np.array([[0, 0], [0, im[i].shape[0]], [im[i].shape[1], 0], [im[i].shape[1], im[i].shape[0]]])
        print(edges)
        t=[]
        for k in range(len(edges)):
            #print(edges[k])
            #print(H[i])
            t.append(applyHomographyP(edges[k], H[i]) )
            print(t)
        # finf minimux x
        for i in range(len(t)):
            #print(i)
            print(t[i])
            if t[i][0] < minX:
                minX = t[i][0]
            elif t[i][0] > maxX:
                maxX = t[i][0]
            elif t[i][1] < minY:
                minY = t[0][1]
            elif t[i][1] > maxY:
                maxY = t[i][1]

    w = int(maxX - minX)
    h = int(maxY - minY)
    print(w,h)
    Ipano = np.zeros((h*len(im),w*len(im)))
####
    list_of_pix=[]
    for n in range(len(im)):
        for i in range(im[n].shape[0]):
            for j in range(im[n].shape[1]):
                point=(i,j)
                print(point)

                coord = applyHomographyP(point, (H[i]))
                val=im[n][i][j]
                pix=[coord,val]
                list_of_pix.append(pix)
    list_of_pix.sort(pix[0])
    index=0
    for i in range(h):
        for j in range(w):
            if (i,j) ==list_of_pix[index][0]:
                Ipano[i][j]=list_of_pix[index][1]
            else:
                Ipano[i][j]=255
    return Ipano


def generatePanorama():
    img1c = cv2.cvtColor(cv2.imread('backyard2.jpg'), cv2.COLOR_BGR2RGB) / 255
    img1 = cv2.cvtColor(cv2.imread('backyard2.jpg'), cv2.COLOR_BGR2GRAY)
    # img2=cv2.imread("backyard2.jpg", 0)
    img2c = cv2.cvtColor(cv2.imread('backyard1.jpg'), cv2.COLOR_BGR2RGB) / 255
    img2 = cv2.cvtColor(cv2.imread('backyard1.jpg'), cv2.COLOR_BGR2GRAY)

    kp1, desc1 = findFeatures(img1)
    kp2, desc2 = findFeatures(img2)
    pos1, pos2 = matchFeatures(kp1, kp2, desc1, desc2)
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    h,maxInlier = ransacHomography(pos1, pos2,1000,1)

    ##parspctive use
    dst = cv2.warpPerspective(img1c, h, (img2c.shape[1] + img1c.shape[1], img2c.shape[0]))  # img2 left to img1
    #  #dst = cv2.warpPerspective(img1,H,(img2.shape[0],img2.shape[1] + img1.shape[1]))
    plt.subplot(122), plt.imshow(dst), plt.title("Warped Image")
    plt.show()
    plt.figure()
    dst[0:img2c.shape[0], 0:img2c.shape[1]] = img2c  # add bluring
    cv2.imwrite("output.jpg", dst)
    plt.imshow(dst)
    plt.show()
    print(h)
    rH=[]
    rH.append(h)
    out = displayMatches(img1, img2, pos1, pos2, maxInlier)
    data=[img1c,img2c]

    Hs = accumulateHomographies1(rH, math.ceil(len(data) / 2))
    print(Hs)


generatePanorama()