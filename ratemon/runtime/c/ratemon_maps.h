#pragma once
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
  __type(value, struct rm_grant_info);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_rwnd SEC(".maps");

// Learn window scaling factor for each flow.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, uint8_t);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_win_scale SEC(".maps");

// Maps flow to the last time that flow received data. This is used to decide
// whether a flow is idle.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, uint64_t);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_last_data_time_ns SEC(".maps");

// Flows in this map have received a recent keepalive and have not gone idle
// since, so they are considered to be active. A flow's entry is set to 1 on the
// receipt of a keepalive packet and deleted when that flow goes idle.
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, RM_MAX_FLOWS);
  __type(key, struct rm_flow);
  __type(value, int);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} flow_to_keepalive SEC(".maps");

// A ringbuffer of struct rm_flow that have finished their grant and are now
// paused. max_entries is the total ringbuf size in bytes and must be both a
// multiple of the page size (4096) and a power of two. A single struct rm_flow
// is 12 bytes. The max number of entries this ringbuf will need to hold is set
// by the RM_MAX_ACTIVE_FLOWS environment variable and will typically be ~20.
// So 4096 bytes is plenty.
struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 4096 /* bytes, one page */);
  __uint(pinning, LIBBPF_PIN_BY_NAME);
} done_flows SEC(".maps");

#endif /* __RATEMON_MAPS_H */
