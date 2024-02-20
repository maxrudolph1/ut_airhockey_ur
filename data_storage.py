import imageio
import cv2
import time
import rtde
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Received Package data types
# https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/

# Might be useful for robot proptioception recording
# https://github.com/UniversalRobots/RTDE_Python_Client_Library/blob/main/examples/record.py

def save_data():
	cap = cv2.VideoCapture(0)
	con = rtde.RTDE('172.22.22.2', 30004)
	con.connect()
	# rcv = RTDEReceive("172.22.22.2")
	while True:
		ret, image = cap.read()
		timestamp = time.time()
		proptioception = con.receive()
		# x = struct.unpack(’!d’, packet)[0] # alternive: directly parse socket package
		imageio.imwrite("", image)


if __name__ == "__main__":
	save_data()
	
