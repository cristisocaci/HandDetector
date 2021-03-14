import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
# Initiate STAR detector
orb = cv2.ORB_create()


while True:
    ret, img = cap.read()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, outImage = None, color=(255,0,0))
    plt.imshow(img2)
    plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



