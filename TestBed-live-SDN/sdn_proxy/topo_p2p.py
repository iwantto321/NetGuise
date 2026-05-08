from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel

from mininet.node import RemoteController



#!/usr/bin/env python3
from mininet.topo import Topo
from mininet.net  import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli  import CLI
from mininet.log  import setLogLevel

class TransparentSDNTopo(Topo):
    def build(self):
        # === Gateway ===
        s1 = self.addSwitch('s1', protocols='OpenFlow13')

        # === Host + Proxy ===
        # dn: echo server
        dn = self.addHost(
            'dn',
            ip='10.0.0.1/24',
            mac='00:00:00:00:00:01'
        )
        
        # proxy: 10.0.0.254:9000
        proxy = self.addHost(
            'proxy',
            ip='10.0.0.254/24',
            mac='00:00:00:00:00:02'
        )

        # h1: client host
        h1 = self.addHost(
            'h1',
            ip='10.0.0.2/24',
            mac='00:00:00:00:00:03'
        )

        # === Link Configuration ===
        # s1-eth1 <--> dn
        # s1-eth2 <--> proxy
        # s1-eth3 <--> h1
        self.addLink(dn, s1, cls=TCLink, bw=1000, delay='0ms')  # s1-eth1
        self.addLink(proxy, s1, cls=TCLink, bw=1000, delay='0ms')  # s1-eth2
        self.addLink(h1, s1, cls=TCLink, bw=1000, delay='0ms') # s1-eth3

        ##################################################
        #                    Topology Overview:
        #             
        #                       dn (10.0.0.1:8000) 
        #                       | eth1
        #                       s1 
        #                 eth2 /  \ eth3
        #     (10.0.0.254) proxy   h1 (10.0.0.2)
        #                
        ##################################################

def print_port_mapping(net):
    """Prints switch ports & interface mapping"""
    print("\n--- Port Mapping ---")
    for sw in net.switches:
        print(f"Switch {sw.name}:")
        for intf in sw.intfList():
            link = intf.link
            if link:
                intf_peer = link.intf1 if link.intf1 != intf else link.intf2
                print(f"  {intf.name} (port {sw.ports[intf]}) --> {intf_peer.node.name}")
    print("--------------------\n")



def run():
    setLogLevel('info')

    # Create topology and launch the network
    topo = TransparentSDNTopo()
    net  = Mininet(
        topo=topo,
        switch=OVSSwitch,
        controller=lambda name: RemoteController(name, ip='127.0.0.1', port=6653),
        link=TCLink
    )
    net.start()

    # Start echo server
    dn = net.get('dn')
    dn.cmd('chmod +x echo_server.py')
    dn.cmd('python3 echo_server.py &')

    # (Optional) Start proxy
    proxy = net.get('proxy')
    proxy.cmd('chmod +x proxy.py')
    proxy.cmd('python3 proxy.py &')

    print("\n--- Host IP Mapping ---")
    for h in (net.get('dn'), net.get('proxy'), net.get('h1')):
        print(f"  {h.name}: {h.IP()}")
    print("proxy is expected to listen on 10.0.0.254:9000 and forward to dn:8000\n")

    print_port_mapping(net)  # port ↔ interface mappings for s1
    CLI(net)
    net.stop()

if __name__ == '__main__':
    run()
