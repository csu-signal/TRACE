import socket
import requests
import ssl, time, random
import json

HOST = "129.82.138.15"  # The server's hostname or IP address (TARSKI)
PORT = 65431  # The port used by the server 

def run_inference_socket(prompt):
    total_start_time = time.perf_counter()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.settimeout(None) 
        sendData = str.encode(prompt)
        send_start_time = time.perf_counter()
        s.sendall(sendData)
        # Signal end-of-transmit so server can exit recv loop immediately
        s.shutdown(socket.SHUT_WR)
        send_elapsed_ms = (time.perf_counter() - send_start_time) * 1000
        print(f"[Tarski] Prompt sent: {len(sendData)} bytes, {len(prompt)} chars in {send_elapsed_ms:.1f} ms")

        data = bytearray()
        while True:
            try:
                chunk = s.recv(4096)
            except ConnectionResetError:
                print("[TARSKI] Connection lost.")
                break
            except socket.timeout:
                continue
            if not chunk:
                break
            data.extend(chunk)

    received = data.decode("utf-8", errors="ignore")
    total_elapsed_s = time.perf_counter() - total_start_time
    print(f"[Tarski] Response received: {len(received)} chars in {total_elapsed_s:.2f} s")
    return received

with open('C:\\GitHub\\TRACE\\mmdemo\\features\\sensorBaseData.json', 'r') as file:
    data = json.load(file)
    
for i in data:
    #print("prompt\n" + i['prompt'])
    r = run_inference_socket(i['prompt'])
    print("response:" + r)

