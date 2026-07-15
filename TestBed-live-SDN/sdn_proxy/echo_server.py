#!/usr/bin/env python3
import socket


HOST = "0.0.0.0"
PORT = 8000


def serve_connection(connection):
    with connection:
        while True:
            data = connection.recv(4096)
            if not data:
                return
            connection.sendall(data)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(16)
        print(f"echo server listening on {HOST}:{PORT}")
        while True:
            connection, address = server.accept()
            print(f"accepted connection from {address}")
            serve_connection(connection)


if __name__ == "__main__":
    main()
