import numpy as np
import argparse
import cv2

def main():
	cap = cv2.VideoCapture(0)

	# Continue capturing webcam until q is pressed
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Blur the frame and change to grayscale
		frame_blurred = cv2.GaussianBlur(frame, (7, 7), 0)
		frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)

		# Create a mask
		lower = np.array([0])
		upper = np.array([80])
		frame_mask = cv2.inRange(frame_gray, lower, upper)

		# Find contours
		_, contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Draw on frame
		# draw buffer
		db = 10

		for contour in contours:
			area = cv2.contourArea(contour)

			if area > 500 and area < 3000:
				cv2.drawContours(frame, [contour], 0, (0,0,255), 2)
				(x,y,w,h) = cv2.boundingRect(contour)
				cv2.rectangle(frame, (x-db, y-db), (x+w+db, y+h+db), (0,255,0), 2)

		# Display the resulting frame
		cv2.imshow('Webcam', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything is done, release the capture
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__': main()
