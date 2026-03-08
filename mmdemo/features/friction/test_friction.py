import socket

HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
PORT = 65432  # The port used by the server

while True:
    transcript = input("Enter transcription: ")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST,PORT))
            sendData = str.encode(transcript)
            print("Send Data Length:" + str(len(sendData)))
            s.sendall(sendData)
            print("Waiting for friction server response")
            data = s.recv(2048)
        received = data.decode()
        # if received != "No Friction":
        friction = received
        print(f"Received from Server:{received}")
    except Exception as e:
        friction = ''
        print(f"FRICTION FEATURE: An error occurred: {e}")
