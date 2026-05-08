#!/usr/bin/env python3
import socket
import threading

# Config
LISTEN_HOST = '0.0.0.0'    # proxy monitoring IP (localhost = all interface)
LISTEN_PORT = 9000         # proxy monitoring port
REMOTE_HOST = '10.0.0.1'   # dst server IP
REMOTE_PORT = 8000         # dst server service port 

TARGET_LEN = 256 # padding/fragment target length (byte)
READ_BUF   = 4096  # Max socket read buffer (byte)


def modify_and_pad(data: bytes) -> bytes:
    """
    Pad or truncate data to TARGET_LEN:
    - If the length is less than TARGET_LEN: add 0x00 at the end
    - If the length is greater than TARGET_LEN: truncate to TARGET_LEN
    """
    if len(data) < TARGET_LEN:
        return data + b'\x00' * (TARGET_LEN - len(data))
    else:
        return data[:TARGET_LEN]

def forward(src_sock: socket.socket, dst_sock: socket.socket, do_modify: bool):
    """
    Read data from src_sock and write to dst_sock.
    If EOF is read (recv returns NULL), shutdown(dst_sock, SHUT_WR) before exiting the thread.
    This ensures that the thread in the other direction has a chance to push the remaining data.
    """
    try:
        while True:
            chunk = src_sock.recv(READ_BUF)
            if not chunk:
                # when src no more msg comes, tell dst_sock no more to sent
                try:
                    dst_sock.shutdown(socket.SHUT_WR)
                except Exception:
                    pass
                break

            if do_modify:
                chunk = modify_and_pad(chunk)
            dst_sock.sendall(chunk)
    except Exception as e:
        # debug
        print(f"[forward] exception: {e}")
    finally:
        # shutdown
        try:
            src_sock.shutdown(socket.SHUT_RD)
        except Exception:
            pass
        src_sock.close()

def handle_connection(client_sock: socket.socket):
    """
    After accept() to a client_sock, a connection to the dst server is established here 
    - two-way forwarding thread is started, 
    - finally join and close resources.
    """
    try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect((REMOTE_HOST, REMOTE_PORT))
    except Exception as e:
        print(f"[handle_connection] FAIL TO CONNECT TO Server: {e}")
        client_sock.close()
        return


    FLAG_MANIPULATE = False  # Flag for forward manipulation 

    # Forward/Uplink：client → server
    t1 = threading.Thread(
        target=forward,
        args=(client_sock, server_sock, FLAG_MANIPULATE),
        daemon=False
    )
    # Backward/Downlink：server → client，transparent transmission 透传 
    t2 = threading.Thread(
        target=forward,
        args=(server_sock, client_sock, False),
        daemon=False
    )
    t1.start()
    t2.start()

    # pending for two threads finished
    t1.join()
    t2.join()

    # close after two threads end
    try:
        server_sock.close()
    except Exception:
        pass
    try:
        client_sock.close()
    except Exception:
        pass

def main():
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind((LISTEN_HOST, LISTEN_PORT))
    listen_sock.listen(5)
    print(f"TCP Proxy listening on {LISTEN_HOST}:{LISTEN_PORT}, forwarding to {REMOTE_HOST}:{REMOTE_PORT}", flush=True)

    try:
        while True:
            client_sock, addr = listen_sock.accept()
            print(f"[main] Accepted connection from {addr}", flush=True)
            threading.Thread(
                target=handle_connection,
                args=(client_sock,),
                daemon=True
            ).start()
    except KeyboardInterrupt:
        print("[main] Shutting down proxy.", flush=True)
    finally:
        listen_sock.close()

if __name__ == '__main__':
    main()



