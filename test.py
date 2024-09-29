
import cv2
cap = cv2.VideoCapture('test.mp4')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('window-name', frame)
    cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    while cv2.waitKey(10) & 0xFF != ord('q'):
       pass 


cap.release()
cv2.destroyAllWindows() # destroy all opened windows
