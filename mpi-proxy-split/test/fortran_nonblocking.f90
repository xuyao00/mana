!****************************************************************************
!*   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
!*   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
!*                                                                          *
!*  This file is part of DMTCP.                                             *
!*                                                                          *
!*  DMTCP is free software: you can redistribute it and/or                  *
!*  modify it under the terms of the GNU Lesser General Public License as   *
!*  published by the Free Software Foundation, either version 3 of the      *
!*  License, or (at your option) any later version.                         *
!*                                                                          *
!*  DMTCP is distributed in the hope that it will be useful,                *
!*  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
!*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
!*  GNU Lesser General Public License for more details.                     *
!*                                                                          *
!*  You should have received a copy of the GNU Lesser General Public        *
!*  License in the files COPYING and COPYING.LESSER.  If not, see           *
!*  <http://www.gnu.org/licenses/>.                                         *
!*****************************************************************************

! Fortran non-blocking calls between two ranks.  Each iteration: an exchange
! with MPI_IRECV and MPI_ISEND completed by MPI_WAITALL with a statuses
! array, one completed by MPI_WAITALL with MPI_STATUSES_IGNORE (which the MPI
! library must not write to), an MPI_IBARRIER completed by MPI_WAIT, and an
! MPI_ALLREDUCE.  One rank waits before it sends or enters the barrier, so
! that requests are pending when a checkpoint comes.  Runs until killed, or
! for '-n ITERATIONS'; then rank 0 prints "fortran_nonblocking: PASS".

program fortran_nonblocking
  use iso_c_binding, only: c_int
  implicit none
  include 'mpif.h'

  interface
    integer(c_int) function usleep(usec) bind(c, name='usleep')
      import :: c_int
      integer(c_int), value :: usec
    end function usleep
  end interface

  integer, parameter :: n = 100
  integer :: ierr, rank, nprocs, other, it, iterations, i, r, m, count
  integer :: sbuf(n), rbuf(n), req(2), breq
  integer :: statuses(MPI_STATUS_SIZE, 2), status(MPI_STATUS_SIZE)
  integer :: ignore(MPI_STATUS_SIZE)
  character(len=32) :: arg

  it = 0
  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
  if (nprocs /= 2) call fail('needs 2 ranks')
  other = 1 - rank
  ignore = MPI_STATUSES_IGNORE(:, 1)  ! A wait must not change it.

  iterations = -1
  do i = 1, command_argument_count() - 1
    call get_command_argument(i, arg)
    if (arg == '-n') then
      call get_command_argument(i + 1, arg)
      read (arg, *) iterations
    end if
  end do

  do while (iterations < 0 .or. it < iterations)
    ! Exchange 1: m words, completed with statuses.
    m = n / 2 + mod(it, n / 2)
    rbuf = -1
    statuses = -1
    call MPI_IRECV(rbuf, n, MPI_INTEGER, other, 1, MPI_COMM_WORLD, req(1), &
                   ierr)
    if (mod(it, 2) == rank) r = usleep(1000_c_int)
    do i = 1, m
      sbuf(i) = val(rank, it, i)
    end do
    call MPI_ISEND(sbuf, m, MPI_INTEGER, other, 1, MPI_COMM_WORLD, req(2), &
                   ierr)
    call MPI_WAITALL(2, req, statuses, ierr)
    if (ierr /= MPI_SUCCESS) call fail('MPI_WAITALL failed (exchange 1)')
    if (any(req /= MPI_REQUEST_NULL)) call fail('requests not freed')
    call MPI_GET_COUNT(statuses(:, 1), MPI_INTEGER, count, ierr)
    if (statuses(MPI_SOURCE, 1) /= other .or. &
        statuses(MPI_TAG, 1) /= 1 .or. count /= m) &
      call fail('wrong status (exchange 1)')
    do i = 1, n
      if ((i <= m .and. rbuf(i) /= val(other, it, i)) .or. &
          (i > m .and. rbuf(i) /= -1)) call fail('wrong value (exchange 1)')
    end do

    ! Exchange 2: the send is posted first; MPI_STATUSES_IGNORE.
    do i = 1, n
      sbuf(i) = val(rank, it, n + i)
    end do
    rbuf = -1
    call MPI_ISEND(sbuf, n, MPI_INTEGER, other, 2, MPI_COMM_WORLD, req(1), &
                   ierr)
    if (mod(it + 1, 2) == rank) r = usleep(1000_c_int)
    call MPI_IRECV(rbuf, n, MPI_INTEGER, other, 2, MPI_COMM_WORLD, req(2), &
                   ierr)
    call MPI_WAITALL(2, req, MPI_STATUSES_IGNORE, ierr)
    if (ierr /= MPI_SUCCESS) call fail('MPI_WAITALL failed (exchange 2)')
    if (any(req /= MPI_REQUEST_NULL)) call fail('requests not freed')
    if (any(MPI_STATUSES_IGNORE(:, 1) /= ignore)) &
      call fail('MPI_WAITALL wrote to MPI_STATUSES_IGNORE')
    do i = 1, n
      if (rbuf(i) /= val(other, it, n + i)) &
        call fail('wrong value (exchange 2)')
    end do

    ! A barrier that one rank enters late.
    call MPI_IBARRIER(MPI_COMM_WORLD, breq, ierr)
    if (ierr /= MPI_SUCCESS) call fail('MPI_IBARRIER failed')
    if (mod(it, 2) == rank) r = usleep(1000_c_int)
    call MPI_WAIT(breq, status, ierr)
    if (ierr /= MPI_SUCCESS) call fail('MPI_WAIT failed')
    if (breq /= MPI_REQUEST_NULL) call fail('barrier request not freed')

    ! The sum over all ranks.
    do i = 1, n
      sbuf(i) = val(rank, it, 2 * n + i)
    end do
    rbuf = -1
    call MPI_ALLREDUCE(sbuf, rbuf, n, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD, &
                       ierr)
    if (ierr /= MPI_SUCCESS) call fail('MPI_ALLREDUCE failed')
    do i = 1, n
      if (rbuf(i) /= val(0, it, 2 * n + i) + val(1, it, 2 * n + i)) &
        call fail('wrong sum')
    end do
    it = it + 1
  end do

  call MPI_BARRIER(MPI_COMM_WORLD, ierr)
  if (rank == 0) then
    write (*, '(a, i0, a)') 'fortran_nonblocking: PASS (', it, ' iterations)'
    flush (6)
  end if
  call MPI_FINALIZE(ierr)

contains

  ! Like mt_value() in mana_test.h, kept within 32 bits.
  integer function val(r, it, i)
    integer, intent(in) :: r, it, i
    val = r * 1000003 + mod(it, 100000) * 7919 + i
  end function val

  subroutine fail(msg)
    character(len=*), intent(in) :: msg
    integer :: e
    write (0, '(a, i0, a, i0, 2a)') 'fortran_nonblocking: rank ', rank, &
      ': iteration ', it, ': ', msg
    flush (0)
    call MPI_ABORT(MPI_COMM_WORLD, 1, e)
    stop 1  ! MPI_ABORT may return before the job is killed.
  end subroutine fail

end program fortran_nonblocking
