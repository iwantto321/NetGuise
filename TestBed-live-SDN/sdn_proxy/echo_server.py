#!/usr/bin/env python3
import socket

HOST = ''    # listen on all interfaces at localhost
PORT = 8000  # service port

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen()
    print(f"[dn] Echo server listening on port {PORT}")
    while True:
        conn, addr = srv.accept()
        print(f"[dn] Connected by {addr}")
        with conn:
            try:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        # Client has closed the write end of the connection
                        break
                    print(f"[dn] Received: {data!r}, echoing back")
                    conn.sendall(data)
            finally:
                # Initiate TCP FIN to ensure a clean 4-way termination
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    # If the socket is already half-closed, ignore the error
                    pass
        # conn.close() is automatically called here by the context manager, after "with conn"
        print(f"[dn] Connection with {addr} closed")
