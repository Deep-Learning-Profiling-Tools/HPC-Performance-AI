/* hpcperf_roi_fortran.c -- C entry points behind the Fortran module in hpcperf_roi.f90.
 * Compile with any C compiler and -I<repo>/tools/timing/roi, link into the program.
 * Same semantics as the macros in hpcperf_roi.h (default OFF, see that header). */
#include "hpcperf_roi.h"

void hpcperf_roi_begin_c(void) { HPCPERF_ROI_BEGIN(); }
void hpcperf_roi_end_c(void) { HPCPERF_ROI_END(); }
void hpcperf_roi_begin_sync_c(void) { HPCPERF_ROI_BEGIN_SYNC(); }
void hpcperf_roi_end_sync_c(void) { HPCPERF_ROI_END_SYNC(); }
void hpcperf_roi_exclude_begin_c(void) { HPCPERF_ROI_EXCLUDE_BEGIN(); }
void hpcperf_roi_exclude_begin_sync_c(void) { HPCPERF_ROI_EXCLUDE_BEGIN_SYNC(); }
void hpcperf_roi_exclude_end_c(void) { HPCPERF_ROI_EXCLUDE_END(); }
