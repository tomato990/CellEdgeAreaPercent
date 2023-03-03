import numpy as np
import cv2
from PIL import Image,ImageOps


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


#path = "drive-download-20230228T000636Z-001/ER-mito"
#for i in [24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 34, 36]:
#path = "drive-download-20230228T000636Z-001/mfn2ko"
#for i in range(9, 27):
#path = "drive-download-20230228T000636Z-001/mfn2ko Resuce"
#for i in range(32,45):
path = "drive-download-20230228T000636Z-001/WT"
for i in [1,2,3,4,5,6,7,8]:
    #skeleton = cv2.imread(path + '/Image ' + str(i) + ' - skeleton.tif')
    #skeleton = cv2.bitwise_not(skeleton)
    img = cv2.imread(path + '/Image ' + str(i) + ' - edges.tif')

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(imggray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_scaled = [scale_contour(cnt, 0.8) for cnt in contours]
    # img = cv2.drawContours(img, contours, -1, (0,255,0), 2)
    img = cv2.drawContours(img, cnt_scaled, -1, (0, 0, 0), -1)
    mask = cv2.bitwise_not(img)



    #cv2.imshow('img', mask)
    #cv2.waitKey()
    cv2.imwrite(path + '/Image ' + str(i) + ' - mask.tif',mask)

    mask = Image.open(path + '/Image ' + str(i)+' - mask.tif')
    mask = ImageOps.grayscale(mask)


    skeleton = Image.open(path + '/Image ' + str(i)+' - skeleton.tif')
    skeleton = ImageOps.invert(skeleton)
    blackOriginal = sum(list(skeleton.getdata()).count(j) for j in range(128))


    mask.paste(skeleton,(0,0),skeleton)
    blackNew = sum(list(mask.getdata()).count(i) for i in range(128))
    #mask.show()
    print(blackNew/blackOriginal)



