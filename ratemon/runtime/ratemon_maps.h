/* SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause) */

#ifndef __RATEMON_MAPS_H
#define __RATEMON_MAPS_H

#include <bpf/bpf_helpers.h>

#include "ratemon.h"

#define PIN_GLOBAL_NS		2


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

#endif /* __RATEMON_MAPS_H */
