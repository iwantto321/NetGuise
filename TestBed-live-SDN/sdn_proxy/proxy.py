#!/usr/bin/env python3
from __future__ import annotations

import socket
import threading


LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9000
REMOTE_HOST = "10.0.0.1"
REMOTE_PORT = 8000
READ_BUFFER = 4096


def forward(source: socket.socket, destination: socket.socket) -> None:
    """Forward a byte stream without changing application data."""

    try:
        while True:
            chunk = source.recv(READ_BUFFER)
            if not chunk:
                try:
                    destination.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            destination.sendall(chunk)
    except OSError as error:
        print(f"forwarding stopped: {error}")
    finally:
        try:
            source.shutdown(socket.SHUT_RD)
        except OSError:
            pass


def handle_connection(client: socket.socket) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.connect((REMOTE_HOST, REMOTE_PORT))
    except OSError as error:
        print(f"could not connect to server: {error}")
        client.close()
        server.close()
        return

    upstream = threading.Thread(target=forward, args=(client, server))
    downstream = threading.Thread(target=forward, args=(server, client))
    upstream.start()
    downstream.start()
    upstream.join()
    downstream.join()
    client.close()
    server.close()


def main() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((LISTEN_HOST, LISTEN_PORT))
    listener.listen(16)
    print(
        f"proxy listening on {LISTEN_HOST}:{LISTEN_PORT}, "
        f"forwarding to {REMOTE_HOST}:{REMOTE_PORT}",
        flush=True,
    )
    try:
        while True:
            client, address = listener.accept()
            print(f"accepted connection from {address}", flush=True)
            threading.Thread(
                target=handle_connection, args=(client,), daemon=True
            ).start()
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()


if __name__ == "__main__":
    main()
