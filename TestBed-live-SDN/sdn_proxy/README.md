# Transparent SDN proxy

The topology connects a client, transparent proxy, and echo server through an
OpenFlow 1.3 switch. The Ryu controller installs DNAT, SNAT, ARP, and forwarding
rules so traffic addressed to the server can traverse the proxy.

```text
server 10.0.0.1:8000
          |
         s1
        /  \
proxy       client
10.0.0.254  10.0.0.2
:9000
```

## Requirements

- Linux
- Python 3
- Mininet
- Open vSwitch
- Ryu

## Run

Start the controller:

```bash
cd TestBed-live-SDN/sdn_proxy
ryu-manager controller.py
```

Start the topology in another terminal:

```bash
sudo python3 topo_p2p.py
```

From the Mininet CLI, open the relevant hosts and start the echo server, proxy,
and client. The proxy intentionally forwards bytes unchanged. A tunnel-layer
shaper can consume actions from `netguise.online.CausalMorpherSession`, but it
must define paired encapsulation and decapsulation before padding is enabled.
