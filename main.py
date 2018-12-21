import numpy as np
import argparse
import cv2
from neuralnetwork import NeuralNetwork
import os, sys

def main():
	input_nodes = 784
	hidden_nodes = 100
	output_nodes = 10
	learning_rate = 0.3
	nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

	cap = cv2.VideoCapture(0)

	parser = argparse.ArgumentParser()
	parser.add_argument('-t', "--trainingfile", help='path to the file to use to train the neural network', type=str)
	args = parser.parse_args()

	print(args.trainingfile)
	if not os.path.isfile(args.trainingfile):
		sys.exit('Training file does not exist.')

	with open(args.trainingfile, 'r') as fp:
		line = fp.readline()
		while line:
			# Create a list from the line
			line = line.split(',')
			# Create an inputs array filled with all numbers except the first one (which is the target)
			inputs = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01
			# Create a targets array that is filled with 0.01 (0 is techinically unreachable by our nn)
			targets = np.zeros(output_nodes) + 0.01
			# Set the target index to 0.99 (1 is technically unreachable by our nn)
			targets[int(line[0])] = 0.99
			# Train with these inputs and targets
			nn.train(inputs, targets)

			line = fp.readline()


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
		db = 2

		for contour in contours:
			area = cv2.contourArea(contour)

			if area > 100:
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
