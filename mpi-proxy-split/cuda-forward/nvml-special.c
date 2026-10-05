/*
 * Hand-written functions of the libnvidia-ml.so.1 shim (the others are
 * generated trampolines).  The CUDA driver and NVML exchange private tables,
 * so a process must have one NVML, the upper half's (as for libcuda).
 */
#include <stddef.h>

#include "fwd-runtime.h"

typedef int nvmlReturn_t;

static void *
upper_sym(const char *symbol)
{
  void *p = NULL;
  FWD_CALL_UPPER(p, NULL, mana_fwd_ctl.uh_dlsym("libnvidia-ml.so.1", symbol));
  return p;
}

nvmlReturn_t
nvmlInternalGetExportTable(const void **ppExportTable, const void *pExportTableId)
{
  static nvmlReturn_t (*real)(const void **, const void *) = NULL;
  if (real == NULL) {
    real = upper_sym("nvmlInternalGetExportTable");
    if (real == NULL) {
      return 1;                         // NVML_ERROR_UNINITIALIZED
    }
  }
  const void *t = NULL;
  nvmlReturn_t rc;
  FWD_CALL_UPPER(rc, 1, real(&t, pExportTableId));
  if (rc == 0 && t != NULL) {
    t = mana_fwd_wrap_table(t, (void *)real, pExportTableId);
  }
  *ppExportTable = t;
  return rc;
}
