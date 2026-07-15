#!/usr/bin/env python3
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo


class TransparentSDNTopo(Topo):
    def build(self):
        switch = self.addSwitch("s1", protocols="OpenFlow13")
        server = self.addHost(
            "dn", ip="10.0.0.1/24", mac="00:00:00:00:00:01"
        )
        proxy = self.addHost(
            "proxy", ip="10.0.0.254/24", mac="00:00:00:00:00:02"
        )
        client = self.addHost(
            "h1", ip="10.0.0.2/24", mac="00:00:00:00:00:03"
        )
        for host in (server, proxy, client):
            self.addLink(host, switch, cls=TCLink, bw=1000, delay="0ms")


def print_port_mapping(network):
    for switch in network.switches:
        print(f"switch {switch.name}")
        for interface in switch.intfList():
            if interface.link:
                peer = (
                    interface.link.intf1
                    if interface.link.intf1 != interface
                    else interface.link.intf2
                )
                print(
                    f"  {interface.name} (port {switch.ports[interface]}) "
                    f"-> {peer.node.name}"
                )


def run():
    setLogLevel("info")
    network = Mininet(
        topo=TransparentSDNTopo(),
        switch=OVSSwitch,
        controller=lambda name: RemoteController(
            name, ip="127.0.0.1", port=6653
        ),
        link=TCLink,
    )
    network.start()
    server = network.get("dn")
    proxy = network.get("proxy")
    server.cmd("python3 echo_server.py &")
    proxy.cmd("python3 proxy.py &")
    for host in (server, proxy, network.get("h1")):
        print(f"{host.name}: {host.IP()}")
    print_port_mapping(network)
    CLI(network)
    network.stop()


if __name__ == "__main__":
    run()
