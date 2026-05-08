#!/usr/bin/env python3
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3, ether
from ryu.lib.packet import packet, ethernet, arp


class TransparentProxy(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    # === Network Parameters ===
    REAL_SERVER_IP   = '10.0.0.1'            # IP of the real backend server
    REAL_SERVER_MAC  = '00:00:00:00:00:01'   # MAC address of the backend server
    PROXY_IP         = '10.0.0.254'          # Transparent proxy IP
    PROXY_MAC        = '00:00:00:00:00:02'   # MAC address of the proxy
    REAL_SERVER_PORT = 8000                 # TCP port the server listens on
    PROXY_TCP_PORT   = 9000                 # TCP port the proxy listens on

    # === Switch Port Mapping (s1) ===
    PORT_TO_SERVER = 1  # s1-eth1 (to real server)
    PORT_TO_PROXY  = 2  # s1-eth2 (to proxy)
    PORT_TO_CLIENT = 3  # s1-eth3 (to client, h1)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle initial switch setup: install static flow rules"""
        dp     = ev.msg.datapath
        ofp    = dp.ofproto
        parser = dp.ofproto_parser

        # (0) Intercept ARP requests from client to real server — proxy ARP
        match = parser.OFPMatch(
            in_port=self.PORT_TO_CLIENT,
            eth_type=ether.ETH_TYPE_ARP,
            arp_tpa=self.REAL_SERVER_IP
        )
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, priority=200, match=match, actions=actions)

        # (1) Flood all other ARP packets
        match = parser.OFPMatch(eth_type=ether.ETH_TYPE_ARP)
        actions = [parser.OFPActionOutput(ofp.OFPP_FLOOD)]
        self.add_flow(dp, priority=100, match=match, actions=actions)

        # (2) ICMP destined to real server — forward to server
        match = parser.OFPMatch(eth_type=0x0800, ip_proto=1, ipv4_dst=self.REAL_SERVER_IP)
        actions = [parser.OFPActionOutput(self.PORT_TO_SERVER)]
        self.add_flow(dp, priority=10, match=match, actions=actions)

        # (3) ICMP replies from server — forward to client
        match = parser.OFPMatch(eth_type=0x0800, ip_proto=1, ipv4_src=self.REAL_SERVER_IP)
        actions = [parser.OFPActionOutput(self.PORT_TO_CLIENT)]
        self.add_flow(dp, priority=10, match=match, actions=actions)

        # (4) DNAT: client → real_server:8000 → proxy:9000
        match = parser.OFPMatch(
            in_port=self.PORT_TO_CLIENT,
            eth_type=0x0800, ip_proto=6,
            ipv4_dst=self.REAL_SERVER_IP, tcp_dst=self.REAL_SERVER_PORT
        )
        actions = [
            parser.OFPActionSetField(eth_dst=self.PROXY_MAC),
            parser.OFPActionSetField(ipv4_dst=self.PROXY_IP),
            parser.OFPActionSetField(tcp_dst=self.PROXY_TCP_PORT),
            parser.OFPActionOutput(self.PORT_TO_PROXY),
        ]
        self.add_flow(dp, priority=200, match=match, actions=actions)

        # (5) Forward all TCP to proxy (proxy-originated or redirected)
        match = parser.OFPMatch(eth_type=0x0800, ip_proto=6, ipv4_dst=self.PROXY_IP)
        actions = [parser.OFPActionOutput(self.PORT_TO_PROXY)]
        self.add_flow(dp, priority=150, match=match, actions=actions)

        # (6) Proxy → real_server (e.g., 9000 → 8000 connection)
        match = parser.OFPMatch(
            in_port=self.PORT_TO_PROXY,
            eth_type=0x0800, ip_proto=6,
            ipv4_dst=self.REAL_SERVER_IP, tcp_dst=self.REAL_SERVER_PORT
        )
        actions = [
            parser.OFPActionSetField(eth_dst=self.REAL_SERVER_MAC),
            parser.OFPActionOutput(self.PORT_TO_SERVER),
        ]
        self.add_flow(dp, priority=150, match=match, actions=actions)

        # (7) Server replies → client (e.g., 8000 → client)
        match = parser.OFPMatch(
            in_port=self.PORT_TO_SERVER,
            eth_type=0x0800, ip_proto=6,
            ipv4_src=self.REAL_SERVER_IP, tcp_src=self.REAL_SERVER_PORT
        )
        actions = [parser.OFPActionOutput(self.PORT_TO_CLIENT)]
        self.add_flow(dp, priority=90, match=match, actions=actions)

        # (8) SNAT: proxy → client (e.g., 9000 → client, fake source as server)
        match = parser.OFPMatch(
            in_port=self.PORT_TO_PROXY,
            eth_type=0x0800, ip_proto=6,
            ipv4_src=self.PROXY_IP, tcp_src=self.PROXY_TCP_PORT
        )
        actions = [
            parser.OFPActionSetField(ipv4_src=self.REAL_SERVER_IP),
            parser.OFPActionSetField(tcp_src=self.REAL_SERVER_PORT),
            parser.OFPActionSetField(eth_src=self.REAL_SERVER_MAC),
            parser.OFPActionOutput(self.PORT_TO_CLIENT),
        ]
        self.add_flow(dp, priority=80, match=match, actions=actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _arp_proxy(self, ev):
        """
        Respond to ARP request for the real server IP from client with a
        proxy ARP reply using the proxy's MAC address.
        """
        msg     = ev.msg
        dp      = msg.datapath
        ofp     = dp.ofproto
        parser  = dp.ofproto_parser
        pkt     = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_req = pkt.get_protocol(arp.arp)
        in_port = msg.match['in_port']

        if arp_req and arp_req.opcode == arp.ARP_REQUEST \
           and in_port == self.PORT_TO_CLIENT \
           and arp_req.dst_ip == self.REAL_SERVER_IP:

            # Construct ARP reply with proxy MAC pretending to be the real server
            eth_reply = ethernet.ethernet(
                dst=eth_pkt.src,
                src=self.PROXY_MAC,
                ethertype=ether.ETH_TYPE_ARP
            )
            arp_reply = arp.arp(
                opcode=arp.ARP_REPLY,
                src_mac=self.PROXY_MAC,
                src_ip=self.REAL_SERVER_IP,
                dst_mac=arp_req.src_mac,
                dst_ip=arp_req.src_ip
            )
            reply = packet.Packet()
            reply.add_protocol(eth_reply)
            reply.add_protocol(arp_reply)
            reply.serialize()

            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=ofp.OFP_NO_BUFFER,
                in_port=ofp.OFPP_CONTROLLER,
                actions=[parser.OFPActionOutput(in_port)],
                data=reply.data
            )
            dp.send_msg(out)

    def add_flow(self, dp, priority, match, actions):
        """Helper method to add a flow to the switch"""
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        instructions = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=instructions
        )
        dp.send_msg(mod)
