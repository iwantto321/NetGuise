#!/usr/bin/env python3
import socket
import sys
import time

DN_IP   = '10.0.0.1'
DN_PORT = 8000
TIMEOUT = 5  # Timeout for waiting for target server's FIN (in seconds)
IAT = 1.0  # Inter-arrival time between messages (seconds)

speech = {
  "messages": [
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.",
    "We are met on a great battlefield of that war.",
    "We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live.",
    "It is altogether fitting and proper that we should do this.",
    "But, in a larger sense, we cannot dedicate—we cannot consecrate—we cannot hallow—this ground.",
    "The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract.",
    "The world will little note, nor long remember what we say here, but it can never forget what they did here.",
    "It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced.",
    "It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion.",
    "That we here highly resolve that these dead shall not have died in vain.",
    "That this nation, under God, shall have a new birth of freedom.",
    "And that government of the people, by the people, for the people, shall not perish from the earth."
  ]
}

messages = speech["messages"]


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cli:
    cli.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reuse of local address (TIME_WAIT state)
    cli.bind(('10.0.0.2', 0)) # Bind client to IP and port (0 for random ephemeral port)
    src_ip, src_port = cli.getsockname()
    print(f"[h1] client {src_ip}:{src_port} → server {DN_IP}:{DN_PORT}")

    try:
        cli.connect((DN_IP, DN_PORT))  # Establish TCP connection to DN server
    except Exception as e:
        print(f"[h1] connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Send each line of the speech, wait for response, then delay
    for idx, msg in enumerate(messages, 1):
        print(f"[h1-sent] msg-{idx}： {msg!r}")
        cli.sendall(msg.encode())  # Send encoded string message to server
        try:
            data = cli.recv(4096)  # Wait for echo response from serve
            print(f"[h1-rec] msg-{idx}： {data.decode()!r}")
            print(f"*************************************************")
            
        except socket.timeout:
            pass
        time.sleep(IAT)  # Wait before sending the next message

    cli.shutdown(socket.SHUT_WR) # Close connetction (sends FIN)
    cli.settimeout(TIMEOUT) # Timeout to avoid forever stuck here    
    while True:
        # one more time recv()，until receive b''（FIN），or surpass TIMEOUT 
        try:
            more = cli.recv(4096)
        except socket.timeout:
            print(f"[h1] Timeout for waiting for FIN（{TIMEOUT}s），Force to Close", file=sys.stderr)
            break

        if not more:
            # recv FIN → ACK from target
            print("[h1] Received Server Side FIN → ACK, 4 way hand shake finished")
            break

print("[h1] Conection Closed")
