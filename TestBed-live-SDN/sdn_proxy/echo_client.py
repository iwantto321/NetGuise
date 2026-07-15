#!/usr/bin/env python3
import socket
import time


SERVER = ("10.0.0.1", 8000)
CLIENT_IP = "10.0.0.2"
MESSAGES = (b"netguise-one", b"netguise-two", b"netguise-three")
INTERVAL_SECONDS = 1.0


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.bind((CLIENT_IP, 0))
        client.settimeout(5.0)
        client.connect(SERVER)
        for message in MESSAGES:
            client.sendall(message)
            response = client.recv(4096)
            if response != message:
                raise RuntimeError("echo response did not match the request")
            print(response.decode("ascii"))
            time.sleep(INTERVAL_SECONDS)
        client.shutdown(socket.SHUT_WR)


if __name__ == "__main__":
    main()
