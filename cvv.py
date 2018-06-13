import numpy as np
import cv2
import imutils    #for resizing the frame
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVDI')
out = cv2.VideoWriter("object.avi", fourcc, 10.0, (500,480))

while 1:            #starting live video
    ret, frame = cap.read()
    greenl = np.array([29,86,6])   #settimg up the lower limit 
    greenu = np.array([64,255,255]) #setting up the upper limit
    #frame = imutils.resize(frame, width = 600)   #pretty simple, right
    blurred = cv2.GaussianBlur(frame, (11,11), 0)  #blurring operation
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenl, greenu)     #till now, done at aryabhatt
    mask = cv2.erode(mask, None, iterations=2)  #eroding and dialating for 2 iteration
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding the contours
    if imutils.is_cv2():
        cnts = cnts[0]	
    else:
        cnts = cnts[1]
    center = None      #initialize center with 0,0
    if len(cnts)>0:
        c = max(cnts, key = cv2.contourArea)      #reading maximum sized contour
        ((x,y),radius) = cv2.minEnclosingCircle(c)  #determinig the size of the object and accordingly adjust the radius
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) #defining center of circle
        if radius>10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)  #draw contour
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.imshow("frame", frame)
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break
	
cap.release()
out.release()
cv2.destroyAllWindows()
