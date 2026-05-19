#pragma once

#ifndef __RATEMON_UTILS_H
#define __RATEMON_UTILS_H

#include <stdbool.h>
#include <string.h>

#include "ratemon.h"

// ---------------------------------------------------------------------------
// RM_CCA helpers
// ---------------------------------------------------------------------------

// Human-readable accepted values for RM_CCA.
#define RM_CCA_ACCEPTED_VALUES                                                 \
  "\"cubic\", \"dctcp\", \"bpf_cubic\", or \"bpf_dctcp\""

// Normalize RM_CCA values to a BPF struct_ops CCA name.
//
// Accepted input aliases:
//   - "cubic" / "bpf_cubic"  -> RM_BPF_CUBIC
//   - "dctcp" / "bpf_dctcp"  -> RM_BPF_DCTCP
//
// Returns true on success and writes the canonical BPF CCA string to
// *normalized_bpf_cca.
static inline bool rm_cca_to_bpf_name(const char *cca,
                                      const char **normalized_bpf_cca) {
  if (cca == NULL || normalized_bpf_cca == NULL || cca[0] == '\0') {
    return false;
  }

  if (strcmp(cca, "cubic") == 0 || strcmp(cca, RM_BPF_CUBIC) == 0) {
    *normalized_bpf_cca = RM_BPF_CUBIC;
    return true;
  }
  if (strcmp(cca, "dctcp") == 0 || strcmp(cca, RM_BPF_DCTCP) == 0) {
    *normalized_bpf_cca = RM_BPF_DCTCP;
    return true;
  }

  return false;
}

#endif /* __RATEMON_UTILS_H */
