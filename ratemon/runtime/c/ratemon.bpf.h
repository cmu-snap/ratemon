#pragma once
// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause

#ifndef __RATEMON_BPF_H
#define __RATEMON_BPF_H

#include "ratemon.h"

// Comment out the below line to disable verbose BPF logging.
// #define RM_VERBOSE

#ifdef RM_VERBOSE
#define RM_PRINTK(...) bpf_printk(__VA_ARGS__)
#else
#define RM_PRINTK(...)
#endif

#endif /* __RATEMON_BPF_H */