from featureModules.IFeature import *
import mediapipe as mp
from utils import *

# class AsrFeature(IFeature):
    # def __init__(self):
        # try: 
        # connected = []
        # connected.append(False)

        # last_data = []
        # last_data.append(None)

        # client = []
        # client.append(None)

        # #next create a socket object 
        # server_ip='192.168.0.113'
        # server_port=9999
        # s = socket.socket()         
        # print ("Server Socket successfully created")

        # s.bind((server_ip, server_port))         
        # print ("socket binded to %s" %(server_port))
        # print ("Server IP address:", server_ip)

        # # put the socket into listening mode 
        # s.listen(2)     
        # print ("socket is listening")  

    # def processFrame(self, deviceId, bodies, w, h, rotation, translation, cameraMatrix, dist, frame, framergb, depth, blocks, blockStatus):
        #     try:
        #         if(connected[0] == False):
        #             client[0], addr = s.accept()  
        #             print ('Got connection from', addr )
        #             connected[0] = True
        #             client[0].send('Thank you for connecting'.encode())
        #             client[0].setblocking(0)
        #         else:
        #             ready = select.select([client[0]], [], [], 0.1)
        #             if ready[0]:
        #                 data = client[0].recv(1024).decode()

        #                 # Only print the data if it's new
        #                 if data != last_data[0]:
        #                     last_data[0] = data  # Update the last received data 
        #                     print(data)
                            
        #                     # wrapped_text = textwrap.wrap(data, width=50)
        #                     # asrFrame = np.zeros((1080, 1920, 3), dtype = "uint8")

        #                     # for i, line in enumerate(wrapped_text):
        #                     #     cv2.putText(asrFrame, str(line), (75,75 * (1+i)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

        #                     # keyFrame[1] = cv2.resize(asrFrame, (640, 360))
        #     except socket.error as e:
        #         print(e.args[0])