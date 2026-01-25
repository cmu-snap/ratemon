// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

// clang-format off
// vmlinux.h needs to be first.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_endian.h>

#include "ratemon.h"
#include "ratemon.bpf.h"
#include "ratemon_maps.h"
// clang-format on

char LICENSE[] SEC("license") = "Dual BSD/GPL";

inline int min(int val1, int val2) { return val1 < val2 ? val1 : val2; }

inline int max(int val1, int val2) { return val1 > val2 ? val1 : val2; }

// Comparators to handle sequence number wrapping for 32-bit sequence numbers.
// The key is to remember that arithmetic on int32_t is module 2^31.
inline bool before(uint32_t seq1, uint32_t seq2) {
  return (int32_t)(seq1 - seq2) < 0;
}

inline bool after(uint32_t seq2, uint32_t seq1) { return before(seq1, seq2); }

enum {
  ETH_P_IP = 0x0800 /* Internet Protocol packet	*/
};

enum { TC_ACT_OK = 0 };

inline void handle_extra_grant(struct rm_flow *flow,
                               struct rm_grant_info *grant_info,
                               uint32_t extra_grant, uint32_t *rwnd) {
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow with remote port "
            "%u granted an extra %u bytes on top of desired grant of %u bytes",
            flow->remote_port, extra_grant, *rwnd);
  // This may go negative (if the extra grant is more than the remaining
  // data). If it does, then this grant will actually pregrant for the
  // sender's next data, which we cannot escape.
  grant_info->ungranted_bytes -= (int)extra_grant;
  // ALWAYS update rwnd_end_seq when granting so that we NEVER retract a
  // grant.
  grant_info->rwnd_end_seq += extra_grant;
  // If the sender has more pending data, then we can expect the sender to
  // meet (some of) this extra grant immediately, so update
  // grant_end_seq by the amount of expected data. If the sender has
  // no more data, then do not update grant_end_seq.
  grant_info->grant_end_seq +=
      min((int)extra_grant, max(grant_info->ungranted_bytes, 0));
  *rwnd += extra_grant;
}

// Perform RWND tuning at TC egress. If a flow has an entry in flow_to_rwnd,
// then install that value in the advertised window field. Inspired by:
// https://stackoverflow.com/questions/65762365/ebpf-printing-udp-payload-and-source-ip-as-hex
SEC("tc/egress")
int do_rwnd_at_egress(struct __sk_buff *skb) {
  // RM_PRINTK("INFO: do_rwnd_at_egress");
  if (skb == NULL) {
    return TC_ACT_OK;
  }

  // trunk-ignore(clang-tidy/performance-no-int-to-ptr)
  void *data = (void *)(long)skb->data;
  // trunk-ignore(clang-tidy/performance-no-int-to-ptr)
  void *data_end = (void *)(long)skb->data_end;
  // We get the packet starting with the Ethernet header and need to parse the
  // network and transport headers.
  struct ethhdr *eth = data;
  struct iphdr *ip = NULL;
  struct tcphdr *tcp = NULL;

  // Check that the Ethernet header does not extend past the end of the packet.
  if ((void *)eth + sizeof(*eth) > data_end) {
    return TC_ACT_OK;
  }
  // Check that the network header is IP, find its start, and check that it does
  // not extend past the end of the packet.
  if (eth->h_proto != bpf_htons(ETH_P_IP)) {
    return TC_ACT_OK;
  }
  ip = data + sizeof(*eth);
  if ((void *)ip + sizeof(*ip) > data_end) {
    return TC_ACT_OK;
  }
  // Check that the transport header is TCP, find its start, and check that it
  // does not extend past the end of the packet.
  if (ip->protocol != IPPROTO_TCP) {
    return TC_ACT_OK;
  }
  tcp = (void *)ip + sizeof(*ip);
  if ((void *)tcp + sizeof(*tcp) > data_end) {
    return TC_ACT_OK;
  }

  // Look up the RWND value for this flow.
  // Note the difference between bpf_ntohl and bpf_ntohs.
  struct rm_flow flow = {.local_addr = bpf_ntohl(ip->saddr),
                         .remote_addr = bpf_ntohl(ip->daddr),
                         .local_port = bpf_ntohs(tcp->source),
                         .remote_port = bpf_ntohs(tcp->dest)};
  struct rm_grant_info *grant_info = bpf_map_lookup_elem(&flow_to_rwnd, &flow);
  if (grant_info == NULL) {
    return TC_ACT_OK;
  }

  uint8_t *win_scale = bpf_map_lookup_elem(&flow_to_win_scale, &flow);
  if (win_scale == NULL) {
    // We do not know the window scale to use for this flow.
    RM_PRINTK("ERROR: 'do_rwnd_at_egress' flow with local port %u, remote "
              "port %u, no win scale",
              flow.local_port, flow.remote_port);
    return TC_ACT_OK;
  }

  uint32_t ack_seq = bpf_ntohl(tcp->ack_seq);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: ack_seq: %u",
            flow.local_port, flow.remote_port, ack_seq);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: ungranted_bytes: %d",
            flow.local_port, flow.remote_port, grant_info->ungranted_bytes);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
            "grant_info->override_rwnd_bytes: %u",
            flow.local_port, flow.remote_port, grant_info->override_rwnd_bytes);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
            "grant_info->new_grant_bytes: %d",
            flow.local_port, flow.remote_port, grant_info->new_grant_bytes);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
            "grant_info->rwnd_end_seq: %u",
            flow.local_port, flow.remote_port, grant_info->rwnd_end_seq);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
            "grant_info->grant_end_seq: %u",
            flow.local_port, flow.remote_port, grant_info->grant_end_seq);
  RM_PRINTK(
      "INFO: 'do_rwnd_at_egress' flow %u<->%u: grant_info->grant_done: %u",
      flow.local_port, flow.remote_port, grant_info->grant_done);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u: "
            "grant_info->grant_end_buffer_bytes: %d",
            flow.local_port, flow.remote_port,
            grant_info->grant_end_buffer_bytes);

  uint32_t rwnd = 0;
  if (grant_info->override_rwnd_bytes == 0xFFFFFFFF) {
    // Override is 2^32-1, so use grant info.

    if (grant_info->grant_end_seq == 0) {
      // This is the first time we've seen this flow.
      grant_info->grant_end_seq = ack_seq;
    }
    if (grant_info->rwnd_end_seq == 0) {
      // This is the first time we've seen this flow.
      grant_info->rwnd_end_seq = ack_seq;
    }

    // Process a new grant, if available.
    if (grant_info->new_grant_bytes > 0) {
      bool is_pregrant = grant_info->is_pregrant;
      int total_grant = grant_info->new_grant_bytes;
      grant_info->new_grant_bytes = 0;
      grant_info->is_pregrant = false;

      // Step 1: Process the normal grant portion (capped by ungranted_bytes).
      // This updates both rwnd_end_seq and grant_end_seq.
      int normal_grant = min(total_grant, max(grant_info->ungranted_bytes, 0));
      if (normal_grant > 0) {
        grant_info->ungranted_bytes -= normal_grant;
        grant_info->rwnd_end_seq += normal_grant;
        // TODO: This will cause problems if rwnd_end_seq is far ahead due to pregrants.
        // TODO: Maybe: grant_info->grant_end_seq = ack_seq + normal_grant.
        grant_info->grant_end_seq += normal_grant;
        RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u received normal grant "
                  "of %d bytes (ungranted now %d)",
                  flow.local_port, flow.remote_port, normal_grant,
                  grant_info->ungranted_bytes);
      }

      // Step 2: If this is a pregrant, process the pregrant portion.
      // The pregrant is the remainder after the normal grant.
      // This only updates rwnd_end_seq, not grant_end_seq, because the
      // pregrant is for future data that doesn't exist yet.
      if (is_pregrant) {
        int pregrant_portion = total_grant - normal_grant;
        if (pregrant_portion > 0) {
          // ungranted_bytes goes negative - this is intentional for pregrants
          grant_info->ungranted_bytes -= pregrant_portion;
          grant_info->rwnd_end_seq += pregrant_portion;
          // Do NOT update grant_end_seq - we don't expect sender to reach this
          RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u received PREGRANT "
                    "of %d bytes (ungranted now %d)",
                    flow.local_port, flow.remote_port, pregrant_portion,
                    grant_info->ungranted_bytes);
        }
      }
    }
    // Calculate window size using sequence number wrapping.
    if (after(grant_info->rwnd_end_seq, ack_seq)) {
      rwnd = grant_info->rwnd_end_seq - ack_seq;
    } else {
      rwnd = (uint32_t)(grant_info->rwnd_end_seq + (UINT32_MAX - ack_seq) + 1);
    }

    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u has base grant "
              "remaining of %u bytes",
              flow.local_port, flow.remote_port, rwnd);

    // Only apply extra grant handling on the last grant of a burst (when
    // ungranted_bytes <= 0). Skip if we just processed a pregrant, because
    // we don't need to add even more extra grant on top.
    if (!is_pregrant && grant_info->ungranted_bytes <= 0 && rwnd > 0) {
      // If we are supposed to send a nonzero grant, then we should not grant
      // less than one segment (1448B) because otherwise the sender will stall
      // for 200ms. If we are about to do that then grant a little bit extra.
      uint32_t min_grant = 1448U;
      if (rwnd < min_grant) {
        handle_extra_grant(&flow, grant_info, min_grant - rwnd, &rwnd);
      }

      // // We lose all precision less then 1 << win_scale. If rwnd has any bits
      // set
      // // in the last win_scale bits, then grant extra to round up the next
      // bit so
      // // that the last win_scale bits are all 0. This also addresses the
      // situation
      // // where the rwnd is less than 1 << win_scale.
      // uint32_t win_scale_mask = (1U << *win_scale) - 1;
      // uint32_t tail = rwnd & win_scale_mask;
      // if (tail) {
      //   handle_extra_grant(&flow, grant_info, (win_scale_mask + 1) - tail,
      //   &rwnd);
      // }

      if (rwnd < 1 << *win_scale) {
        handle_extra_grant(&flow, grant_info, (1 << *win_scale) - rwnd, &rwnd);
      }

      RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u has %u bytes remaining "
                "in grant",
                flow.local_port, flow.remote_port, rwnd);
    }

    // Check if the grant is over. This check must be grant_end_seq, not
    // rwnd_end_seq. Adjust the end seq by grant_end_buffer_bytes.
    uint32_t actual_grant_end_seq =
        grant_info->grant_end_seq -
        (uint32_t)grant_info->grant_end_buffer_bytes;

    // Debug: log the grant_done check values to understand timing
    bool should_trigger = (ack_seq == actual_grant_end_seq ||
                           after(ack_seq, actual_grant_end_seq));
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u grant_done_check: ack=%u",
              flow.local_port, flow.remote_port, ack_seq);
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u grant_done_check: "
              "actual_end=%u",
              flow.local_port, flow.remote_port, actual_grant_end_seq);
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u grant_done_check: "
              "end-ack=%d",
              flow.local_port, flow.remote_port, (int)actual_grant_end_seq-(int)ack_seq);
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u grant_done_check: "
              "should_trigger=%u ",
              flow.local_port, flow.remote_port, should_trigger);
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u grant_done_check: "
              "already_done=%u",
              flow.local_port, flow.remote_port, grant_info->grant_done);

    if (should_trigger) {
      // Check grant_done so that we only submit this flow to done_flows once.
      if (!grant_info->grant_done) {
        grant_info->grant_done = true;
        // The flow has exhausted its grant, so add it to the done_flows
        // ringbuf.
        RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u exhausted its "
                  "grant, adding to done_flows",
                  flow.local_port, flow.remote_port);
        if (bpf_ringbuf_output(&done_flows, &flow, sizeof(flow), 0)) {
          RM_PRINTK("ERROR: 'do_rwnd_at_egress' error adding flow %u<->%u to "
                    "done_flows",
                    flow.local_port, flow.remote_port);
          return 0;
        }
      }
    }
  } else {
    // Override is not 2^32-1, so ignore grant info and use the override value.
    rwnd = grant_info->override_rwnd_bytes;
    RM_PRINTK("INFO: 'do_rwnd_at_egress' flow %u<->%u has override RWND %u",
              flow.local_port, flow.remote_port, rwnd);
  }

  // Apply the new RWND value.

  // Apply the window scale to the configured RWND value.
  uint16_t rwnd_with_win_scale = (uint16_t)(rwnd >> *win_scale);
  // Set the RWND value in the TCP header. If the existing advertised window
  // set by flow control is smaller, then use that instead so that we
  // preserve flow control.
  uint16_t existing_rwnd_with_win_scale = bpf_ntohs(tcp->window);
  if (existing_rwnd_with_win_scale < rwnd_with_win_scale) {
    RM_PRINTK("WARNING: 'do_rwnd_at_egress' flow with remote port %u existing "
              "advertised window %u smaller than grant %u (values printed do "
              "not include win scale)",
              flow.remote_port, existing_rwnd_with_win_scale,
              rwnd_with_win_scale);
    rwnd_with_win_scale = existing_rwnd_with_win_scale;
  }
  tcp->window = bpf_htons(rwnd_with_win_scale);
  RM_PRINTK("INFO: 'do_rwnd_at_egress' set RWND for flow with remote port %u "
            "to %u (win scale: %u)",
            flow.remote_port, rwnd_with_win_scale, *win_scale);
  return TC_ACT_OK;
}