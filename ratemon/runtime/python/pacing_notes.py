"""This module contains functionality related to packing ACKs."""

#
#
#
##

# - just keep a map of four-tuple to flow object
# Information we need for a flow object:
# - list of incoming packets
# - probably a lock
#   - use one lock for the master flows list and then one lock each for each flow
#   - be careful about holding two locks at once
# - current fairness mitigation state: yes or no
#   - if yes, information about current action: pacing rate, TC filter info, etc.

# tc qdisc add dev lo root handle 1: htb default 30

# tc class add dev lo parent 1:0 classid 1:1 htb rate 1gbit burst 10m

# tc class add dev lo parent 1:1 classid 1:10 htb rate 500mbit burst 10m
# tc class add dev lo parent 1:1 classid 1:20 htb rate 600mbit ceil 900mbit burst 10m
# tc class add dev lo parent 1:1 classid 1:30 htb rate 200mbit ceil 900mbit burst 10m

# tc qdisc add dev lo parent 1:10 handle 10: pfifo
# tc qdisc add dev lo parent 1:20 handle 20: pfifo
# tc qdisc add dev lo parent 1:30 handle 30: pfifo

# tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip src 2.4.6.8/32 match ip sport 3453 match ip dst 1.2.3.4/32 match ip dport 9998 0xffff flowid 1:10
# tc filter add dev lo protocol ip parent 1:0 prio 1 u32 match ip src 2.4.6.8/32 match ip sport 3454 match ip dst 4.3.2.1/32 match ip dport 9999 0xffff flowid 1:20

# An example with htb qdisc, lets assume eth0 == 2::

#     #          u32 -->    +--> htb 1:10 --> sfq 10:0
#     #          |          |
#     #          |          |
#     # eth0 -- htb 1:0 -- htb 1:1
#     #          |          |
#     #          |          |
#     #          u32 -->    +--> htb 1:20 --> sfq 20:0

#     eth0 = 2
#     # add root queue 1:0
#     ip.tc("add", "htb", eth0, 0x10000, default=0x200000)

#     # root class 1:1
#     ip.tc("add-class", "htb", eth0, 0x10001,
#           parent=0x10000,
#           rate="256kbit",
#           burst=1024 * 6)

#     # two branches: 1:10 and 1:20
#     ip.tc("add-class", "htb", eth0, 0x10010,
#           parent=0x10001,
#           rate="192kbit",
#           burst=1024 * 6,
#           prio=1)
#     ip.tc("add-class", "htb", eht0, 0x10020,
#           parent=0x10001,
#           rate="128kbit",
#           burst=1024 * 6,
#           prio=2)

#     # two leaves: 10:0 and 20:0
#     ip.tc("add", "sfq", eth0, 0x100000,
#           parent=0x10010,
#           perturb=10)
#     ip.tc("add", "sfq", eth0, 0x200000,
#           parent=0x10020,
#           perturb=10)

#     # two filters: one to load packets into 1:10 and the
#     # second to 1:20
#     ip.tc("add-filter", "u32", eth0,
#           parent=0x10000,
#           prio=10,
#           protocol=socket.AF_INET,
#           target=0x10010,
#           keys=["0x0006/0x00ff+8", "0x0000/0xffc0+2"])
#     ip.tc("add-filter", "u32", eth0,
#           parent=0x10000,
#           prio=10,
#           protocol=socket.AF_INET,
#           target=0x10020,
#           keys=["0x5/0xf+0", "0x10/0xff+33"])


def add_tc():
    ipr = IPRoute()

    ipr.tc("add", "")
