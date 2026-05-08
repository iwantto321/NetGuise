# Transparent SDN Proxy with Mininet and Ryu

A simple SDN-based transparent proxy architecture using [Mininet](https://github.com/mininet/mininet) and [Ryu](https://osrg.github.io/ryu/), enabling interception, redirection, and transparent forwarding of TCP connections — similar to a userspace man-in-the-middle system.

## 🧠 Overview

This project demonstrates how to transparently redirect and forward TCP traffic (e.g., HTTP) from a client to a proxy, and then forward the proxy’s processed response back to the client — while pretending the traffic is between the client and the original server.

All redirection is handled at the SDN switch using OpenFlow 1.3 rules set by a custom Ryu controller.

## 📡 Topology
                        (10.0.0.1:8000)
                            Server/dn 
                               | eth1
                             +----+
                             | s1 | Gateway (OpenFlow Switch)
                             +----+
                    eth2    /      \     eth3
                       Proxy      Client/h1
            (10.0.0.254:9000)     (10.0.0.2)
### 🧪 Roles

- **`h1`**: Client that initiates TCP connection to `10.0.0.1:8000`
- **`proxy`**: Transparent proxy, receives traffic redirected from h1 and forwards to real server
- **`dn`**: Real server running echo service on port `8000`
- **`s1`**: SDN switch controlled by Ryu, performs DNAT/SNAT/ARP proxying


## ⚙️ Requirements

- Python 3.x
- Mininet
- Ryu SDN Controller
- `ebtables` (optional for L2-level interception)

## 🔧📎📄 Execution
```bash
(terminal-1) ryu-manager controller.py
```

```bash
(terminal-2) sudo python3 topo_p2p.py  # This will give enter the mininet CLI
mininet> xterm h1  # This will enter client CLI
```

```bash
[h1] python3 echo_client.py
```

