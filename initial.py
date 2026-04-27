import cv2
import numpy as np

FILE_PATH = ""

cap = cv2.VideoCapture(FILE_PATH)

while cap.isOpened():
    ret, frame = cap.read() # This reads frame-by-frame the captured VideoCapture

    if not ret:
        print("Video ended")
        break

    # Applying HSV pre-processing to frame to isolate the ball
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create White Color HSV Range
    lower_white = np.array([0,0,200])
    high_white = np.array([180,50,255])

    # Create the White mask on the frame
    mask = cv2.inRange(hsv, lower_white, high_white)

    # Filter out noise with Mathematical Morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find ball contours
    countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter and Draw
    for cnt in countours:
        area = cv2.contourArea(cnt)

        # Area filter to avoid particle detection
        if area > 100:
            # Create the smallest circle possible that wraps the countour
            (x,y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            # Draw the Countour circle
            cv2.circle(frame, center, radius, (0,255,0), 2)

            # Draw a circle on the center
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)


    # User exibhition
    cv2.imshow("Baseball tracker", frame)
    cv2.imshow("Ball Maks", mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




    