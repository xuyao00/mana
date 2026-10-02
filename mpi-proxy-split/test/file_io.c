/****************************************************************************
 *   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
 *   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// MPI-IO on two shared files.  'file_io.dat' stays open from start to end,
// and so across checkpoints and restarts.  Each iteration, each rank writes
// a block of it with MPI_File_write_at, and after a barrier reads back its
// own block, the other rank's, and older blocks of the other rank.  Every
// few iterations, 'file_io2.dat' is opened, written, read back and closed.
//
// The files are not rolled back at a restart, so the ranks may find there
// what they wrote after the checkpoint.  Each block has its own offset and
// content, both set by the iteration, so the ranks write the same data when
// they run an iteration again.

#include "mana_test.h"

#define N 64        // ints in a block
#define REOPEN 10   // 'file_io2.dat' is opened every REOPEN iterations

// The offset of the block of 'rank' in 'slot'.
static MPI_Offset
offset(long slot, int rank)
{
  return ((MPI_Offset)slot * mt_size + rank) * N * sizeof(int);
}

static void
write_block(MPI_File fh, long slot, long it)
{
  int buf[N], count;
  MPI_Status status;
  for (int i = 0; i < N; i++) {
    buf[i] = mt_value(mt_rank, it, i);
  }
  MT_MPI(MPI_File_write_at(fh, offset(slot, mt_rank), buf, N, MPI_INT,
                           &status));
  MPI_Get_count(&status, MPI_INT, &count);
  MT_CHECK(count == N, "iteration %ld: wrote %d ints", it, count);
}

// Checks that 'rank' wrote its block in 'slot' in iteration 'it'.
static void
check_block(MPI_File fh, long slot, int rank, long it, long now)
{
  int buf[N], count;
  MPI_Status status;
  memset(buf, 0xff, sizeof(buf));
  MT_MPI(MPI_File_read_at(fh, offset(slot, rank), buf, N, MPI_INT,
                          &status));
  MPI_Get_count(&status, MPI_INT, &count);
  MT_CHECK(count == N, "iteration %ld: read %d ints of slot %ld", now,
           count, slot);
  for (int i = 0; i < N; i++) {
    MT_CHECK(buf[i] == mt_value(rank, it, i),
             "iteration %ld: slot %ld rank %d word %d: %d", now, slot, rank,
             i, buf[i]);
  }
}

// The size of 'fh' must cover the blocks of 'slots' slots.
static void
check_size(MPI_File fh, long slots, long now)
{
  MPI_Offset size;
  MT_MPI(MPI_File_get_size(fh, &size));
  MT_CHECK(size >= offset(slots, 0), "iteration %ld: size %lld < %lld", now,
           (long long)size, (long long)offset(slots, 0));
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "file_io");
  int other = (mt_rank + 1) % mt_size;
  MPI_File fh, fh2;
  MT_MPI(MPI_File_open(MPI_COMM_WORLD, "file_io.dat",
                       MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh));
  long it;
  for (it = 0; mt_continue(it); it++) {
    if (it % mt_size == mt_rank) {
      usleep(2000);
    }
    write_block(fh, it, it);
    MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
    check_block(fh, it, mt_rank, it, it);
    check_block(fh, it, other, it, it);
    if (it > 0) {
      check_block(fh, it - 1, other, it - 1, it);
      check_block(fh, it / 2, other, it / 2, it);
    }
    check_size(fh, it + 1, it);

    if (it % REOPEN == 0) {
      long session = it / REOPEN;
      MT_MPI(MPI_File_open(MPI_COMM_WORLD, "file_io2.dat",
                           MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL,
                           &fh2));
      write_block(fh2, session, it);
      MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
      check_block(fh2, session, other, it, it);
      if (session > 0) {
        check_block(fh2, session - 1, other, it - REOPEN, it);
      }
      check_size(fh2, session + 1, it);
      MT_MPI(MPI_File_close(&fh2));
      MT_CHECK(fh2 == MPI_FILE_NULL, "iteration %ld: %s", it,
               "MPI_File_close did not set MPI_FILE_NULL");
    }
    usleep(1000);
  }
  MT_MPI(MPI_File_close(&fh));
  mt_finish(it);
  return 0;
}
