### Install sing-box
```
bash <(curl -fsSL https://sing-box.app/deb-install.sh)
```
sing-box [Management commands](https://sing-box.sagernet.org/installation/package-manager/#service-management)

### Config
```
...
  inbounds: [{
      "type": "mixed",
      "tag": "mixed-in",
      "listen": "0.0.0.0",
      "listen_port": 2080,
      "tcp_fast_open": false,
      "sniff": true,
      "sniff_override_destination": false,
      "set_system_proxy": false
    },
    {
      "type": "tun",
      "tag": "tun-in",
      "interface_name": "tun0",
      "inet4_address": "172.18.0.1/30",
      "inet6_address": "fdfe:dcba:9876::1/126",
      "mtu": 9000,
      "gso": false,
      "auto_route": true,
      "strict_route": true,
      "inet4_route_address": ["0.0.0.0/1", "128.0.0.0/1"],
      "inet6_route_address": ["::/1", "8000::/1"],
      "inet4_route_exclude_address": "192.168.0.0/16",
      "inet6_route_exclude_address": "fc00::/7",
      "endpoint_independent_nat": false,
      "udp_timeout": "5m",
      "stack": "gvisor",
      "platform": {
        "http_proxy": {
          "enabled": false,
          "server": "127.0.0.1",
          "server_port": 2080
        }
      },
      "sniff": true,
      "sniff_override_destination": false
    }],
...
  "route": {
    "rule_set": [],
    "rules": [
      {
        "protocol": "dns",
        "outbound": "dns"
      },
      {
        "type": "logical",
        "mode": "or",
        "rules": [
          {
            "domain_regex": "^stun\\..+"
          },
          {
            "domain_keyword": ["stun", "httpdns"]
          },
          {
            "protocol": "stun"
          }
        ],
        "outbound": "block"
      },
      {
        "type": "logical",
        "mode": "or",
        "rules": [
          { "port": [22, 3389] }
        ],
        "outbound": "direct"
      }
    ],
    "final": "Manual Proxy",
    "auto_detect_interface": true
  }
```
Everything will be the same except the `inbounds` and `route` part.

Test: `nc -vz raw.githubusercontent.com 443`
> Connection to raw.githubusercontent.com 443 port [tcp/https] succeeded!
