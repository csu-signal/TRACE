# in pass a dialog history, send every 10 utterances to the friction modal using a sliding window
# and log the output. can be used to test friction outputs in various scenerios

import socket
import csv
from pathlib import Path

HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
PORT = 65432  # The port used by the server
minUtteranceValue = 10
needs_header = True
outputFileName = "withEmbodimentGroup1.csv"
dialogHistory = [
    "it's right there",
    "um it's turned off",
    "ok well red block would be nine minus ten",
    "red block's ten so then",
    "just like put it on",
    "red block, blue block seems pretty balanced",
    "yeah ok so now we know that blue block is also ten",
    "what would we maybe put blue block one on there too",
    "purple block's more",
    "lets try yellow block one oh gee louise green block one",
    "ok so green block one is probably twenty ten ten twenty",
    "ok so now lets start at thirty alright put one of yellow block, purple block on there",
    "probably thirty at this point",
    "ok so we got ok yeah yellow block one is really heavy so lets do the a purple block and a ten",
    "interesting so purple block, red block, blue block's ten twenty",
    "so we said purple block ones thirty i still feel like purple block, red block, blue block's heavier and green block one is twenty",
    "yeah yeah the blue dark blue one is thirty purple block's the larger one purple block, red block, blue block's slightly heavier yeah",
    "we can replace one of red block, blue block with the twenty to get",
    "ok and then put blue block on yellow block one",
    "how much is the dark blue small one ten",
    "ten red block, blue block're both",
    "assuming that this was",
    "oh gosh i think i got them mixed up",
    "yeah im thinking that red block, blue block, green block, yellow block, purple block's ten ten twenty thirty",
    "it's just like slightly",
    "and now it stopped being too heavy",
    "so purple block, green block is supposedly",
    "thirty fourty fifty correct and purple block, green block's still heavier than yellow block thing",
    "so thirty purple block, blue block is also forty right so purple block, blue block if forty yeah",
    "ok so green block one's twenty right",
    "purple block, green block is fifty it's not",
    "yo that's close enough ok so",
    "yeah fifty so we say yellow is fifty",
    "and green is twenty ok so which one is blue and which one is purple",
    "i think that's i think that's good",
    "i think purple block one is purple and blue block one is blue",
    "ok so blue is ten and purple is",
    "yes verify real quick but i think purple block is",
    "yeah we got them yeah",
    "that looks good to me",
    "uh blue yeah so you said purple sweet",
    "purple is thirty red is twenty yellow is fifty"
]

def log(filename, subsetTranscriptions, friction, needs_header):
        file: Path = filename
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            header_row = ["Transcriptions"]
            output_row = [subsetTranscriptions]
            header_row.append("Friction")
            output_row.append(friction)
    

            if needs_header:
                writer.writerow(header_row)
            writer.writerow(output_row)

for i in range(0, len(dialogHistory), 1):
    frictionSubset = dialogHistory[i:i + minUtteranceValue]
    # Process chunk (e.g., print, calculate sum, etc.)
    subsetTranscriptions = ''
    for utter in frictionSubset:
        subsetTranscriptions += utter + "\n"
    print(subsetTranscriptions)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            sendData = str.encode(subsetTranscriptions)
            print("Send Data Length:" + str(len(sendData))) 
            s.sendall(sendData)
            print("Waiting for friction server response")
            data = s.recv(2048)
        received = data.decode()
        if received != "No Friction":
            friction = received
            print(f"Received from Server:{received}")
    except Exception as e:
        friction = ''
        print(f"FRICTION FEATURE THREAD: An error occurred: {e}")

    log(outputFileName, subsetTranscriptions.replace("\n", "\\"), friction, needs_header)
    if needs_header:
        needs_header = False

# while True:
#     transcript = input("Enter transcription: ")
    # try:
    #     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #         s.connect((HOST,PORT))
    #         sendData = str.encode(transcript)
    #         print("Send Data Length:" + str(len(sendData)))
    #         s.sendall(sendData)
    #         print("Waiting for friction server response")
    #         data = s.recv(2048)
    #     received = data.decode()
    #     if received != "No Friction":
    #         friction = received
    #         print(f"Received from Server:{received}")
    # except Exception as e:
    #     friction = ''
    #     print(f"FRICTION FEATURE: An error occurred: {e}")
