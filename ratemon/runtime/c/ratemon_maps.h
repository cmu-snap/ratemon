// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#ifndef __RATEMON_MAPS_H
#define __RATEMON_MAPS_H

// clang-format off
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

#include "ratemon.h"
// clang-format on

// Read RWND limit for flow, as set by userspace. Even though the advertised
// window is only 16 bits in the TCP header, use 32 bits here because we have
// not taken window scaling into account yet.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, unsigned int);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_rwnd SEC(".maps");

// Learn window scaling factor for each flow.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, unsigned char);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_win_scale SEC(".maps");

// Maps flow to the last time that flow received data. This is used to decide
// whether a flow is idle.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, unsigned long);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_last_data_time_ns SEC(".maps");

// Flows in this map have received a recent keepalive and have not gone idle since, so they are considered to be active.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, int);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_keepalive SEC(".maps");

#endif /* __RATEMON_MAPS_H */
