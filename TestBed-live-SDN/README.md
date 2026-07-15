# Controlled SDN testbed

This directory contains the Mininet and Ryu topology used for controlled
end-to-end routing and replay checks. See `sdn_proxy/README.md` for setup.

The transparent proxy forwards application bytes unchanged. NetGuise padding
must be applied by a paired tunnel endpoint with explicit framing; it is not
implemented by appending or truncating arbitrary TCP application payloads.
