! hpcperf_roi.f90 -- Fortran interface to the ROI markers (tools/timing/roi/hpcperf_roi.h).
! Link hpcperf_roi_fortran.c into the program. Default OFF: see hpcperf_roi.h.
!
!   use hpcperf_roi
!   call hpcperf_roi_begin()   ! ... the computation ...   call hpcperf_roi_end()
module hpcperf_roi
  implicit none
  interface
    subroutine hpcperf_roi_begin() bind(C, name="hpcperf_roi_begin_c")
    end subroutine hpcperf_roi_begin
    subroutine hpcperf_roi_end() bind(C, name="hpcperf_roi_end_c")
    end subroutine hpcperf_roi_end
    subroutine hpcperf_roi_begin_sync() bind(C, name="hpcperf_roi_begin_sync_c")
    end subroutine hpcperf_roi_begin_sync
    subroutine hpcperf_roi_end_sync() bind(C, name="hpcperf_roi_end_sync_c")
    end subroutine hpcperf_roi_end_sync
    subroutine hpcperf_roi_exclude_begin() bind(C, name="hpcperf_roi_exclude_begin_c")
    end subroutine hpcperf_roi_exclude_begin
    subroutine hpcperf_roi_exclude_begin_sync() bind(C, name="hpcperf_roi_exclude_begin_sync_c")
    end subroutine hpcperf_roi_exclude_begin_sync
    subroutine hpcperf_roi_exclude_end() bind(C, name="hpcperf_roi_exclude_end_c")
    end subroutine hpcperf_roi_exclude_end
  end interface
end module hpcperf_roi
