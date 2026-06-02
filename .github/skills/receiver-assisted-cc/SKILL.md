---
name: receiver-assisted-cc
description: "Understand and troubleshoot the IBG + RateMon receiver-assisted congestion-control stack. USE FOR: ibg, ratemon, single_request mode, pregrant policy, grant_done behavior, rwnd tuning, tc egress BPF, struct_ops cubic/dctcp, scheduler fairness, burst orchestration, RM_* env/config mapping, experiment interpretation. DO NOT USE FOR: unrelated networking systems or generic TCP tuning outside this codebase."
---

# Receiver-Assisted Congestion Control Skill (IBG + RateMon)

Use this skill when the user asks about how the IBG + RateMon system works, why an experiment behaved a certain way, or where to change behavior in scheduling/grant logic.

## Scope

This system spans two repositories in the same workspace:

- `cctestbed/ibg`: traffic generator and burst orchestration.
- `ratemon/ratemon/runtime/c`: receiver-side scheduler + BPF enforcement.

## Always Start Here

Read these files first before answering architecture or behavior questions:

- `../cctestbed/ibg/receiver.cpp`
- `../cctestbed/ibg/sender.cpp`
- `../cctestbed/ibg/ibg_runner/experiment.py`
- `../cctestbed/ibg/ibg_runner/config.py`
- `../cctestbed/ibg/ibg_runner/defaults.py`
- `../cctestbed/ibg/ibg_runner/ibg_config_*.json` (representative configs)
- `ratemon/runtime/c/libratemon_interp.cpp`
- `ratemon/runtime/c/ratemon_main.c`
- `ratemon/runtime/c/ratemon_tc.bpf.c`
- `ratemon/runtime/c/ratemon_structops_cubic.bpf.c`
- `ratemon/runtime/c/ratemon_structops_dctcp.bpf.c`
- `ratemon/runtime/c/ratemon_structops.bpf.h`
- `ratemon/runtime/c/ratemon_sockops.bpf.c`
- `ratemon/runtime/c/ratemon_kprobe.bpf.c`
- `ratemon/runtime/c/ratemon_maps.h`
- `ratemon/runtime/c/ratemon.h`

## Mental Model

Treat the system as 4 cooperating loops:

1. **Experiment orchestration (Python):**
   - Converts `ibg_runner` config into receiver/sender CLI args and `RM_*` env vars.
   - Starts/stops sender, receiver, and (optionally) ratemon BPF runtime.

2. **Burst control loop (IBG receiver/sender):**
   - Receiver sends burst requests (`rm_burst_request`) to senders.
   - In `single_request` mode: one request per sender host; sender thread 0 broadcasts to all sender threads.

3. **Userspace scheduling loop (libratemon_interp via LD_PRELOAD):**
   - Intercepts `accept/connect/send/close`.
   - Registers monitored flows, sets BPF CCA (`bpf_cubic` or `bpf_dctcp`), tracks active/paused queues.
   - Updates per-flow grant state in BPF maps on each burst request.

4. **Packet enforcement loop (tc egress BPF):**
   - Rewrites advertised RWND per ACK based on `flow_to_rwnd` + `flow_to_win_scale`.
   - Emits grant-complete events through `done_flows` ringbuf.

## Critical Data Structures

- `rm_burst_request` (`ratemon.h`): burst_idx, bytes, scheduled_send_time_ns.
- `rm_grant_info` (`ratemon.h`):
  - `ungranted_bytes`
  - `new_grant_bytes`
  - `rwnd_end_seq`, `grant_end_seq`
  - `grant_done`, `is_pregrant`
  - `grant_end_buffer_bytes`
- Pinned maps (`ratemon_maps.h`):
  - `flow_to_rwnd`
  - `flow_to_win_scale`
  - `flow_to_last_data_time_ns`
  - `flow_to_keepalive`
  - `done_flows` ringbuf

## Modes and Policies

### new_burst_mode

- `normal`: schedule per flow on each request.
- `port`: initial flash grants based on remote-port range.
- `single`: one sender-host request represents all flows on that host.

### single_request_policy (single mode)

- `normal`: schedule on every burst request.
- `pregrant`: schedule first burst normally, then rely on end-of-burst pregrants for overlap.

## Configuration Mapping (must verify)

When diagnosing behavior, confirm this mapping:

- `single_request` (IBG flag) -> receiver/sender request fanout behavior.
- `do_ratemon` -> enables LD_PRELOAD scheduler logic.
- `do_ratemon_bpf` -> enables tc/sockops/struct_ops/kprobe runtime attach.
- `max_active_flows` -> scheduler parallelism cap.
- `epoch_bytes` -> grant chunk size (byte mode).
- `grant_end_buffer_bytes` -> early completion threshold for grant_done.
- `new_burst_mode` + `single_request_policy` -> scheduling policy branch in `libratemon_interp.cpp`.
- `cca` -> struct_ops variant (`bpf_cubic` or `bpf_dctcp`).

## Known Hotspots

- Pregrant correctness:
  - `libratemon_interp.cpp`: `handle_send_single_mode()`, `handle_grant_done()`.
  - `ratemon_tc.bpf.c`: pregrant processing and grant_done emission.
- Spurious or late completion behavior:
  - `grant_end_seq` vs `rwnd_end_seq` handling.
  - `processed_new_grant` guard.
  - ringbuf `done_flows` consumer timing.
- Sender/receiver synchronization in single mode:
  - sender thread-0 broadcast and `threads_ready_count` barrier.

## Troubleshooting Checklist

When user asks "why did throughput collapse/alternate/stall?":

1. Confirm active config values (single_request, new_burst_mode, policy, max_active_flows, epoch_bytes).
2. Verify `do_ratemon` and `do_ratemon_bpf` are both effectively enabled when expected.
3. Inspect receiver/sender logs for burst index progression and synchronization issues.
4. Inspect ratemon logs for:
   - grant_done rate
   - pregrant issuance timing
   - between-bursts warnings
   - BPF experienced error flags
5. Correlate with BPF map dumps (`flow_to_rwnd`, `done_flows`) when available.

## How to Answer User Questions

- Start with the end-to-end control path relevant to the question.
- Anchor claims in exact function names and files.
- Distinguish clearly between:
  - IBG orchestration behavior
  - userspace scheduling behavior
  - BPF packet-level enforcement behavior
- If proposing fixes, state which layer owns the bug and why.

## Non-Goals

Do not generalize to unrelated TCP systems. This skill is specifically for this IBG + RateMon implementation and its experiment runner.
