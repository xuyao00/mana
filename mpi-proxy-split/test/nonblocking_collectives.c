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

// Non-blocking collectives: MPI_Ibarrier, MPI_Ibcast and MPI_Ireduce with a
// rotating root.  Each iteration completes them in another way: MPI_Wait at
// once; MPI_Wait after a sleep, so that a checkpoint finds them pending;
// MPI_Test until done; or all three together with MPI_Waitall.  One rank per
// iteration is late, so the others wait for it to post.

#include "mana_test.h"

#define N 16

enum { WAIT, SLEEP_WAIT, TEST, WAITALL, NMODES };

static long it;

static void
complete(MPI_Request *req, int mode, const char *what)
{
  if (mode == SLEEP_WAIT) {
    usleep(1000 + 1000 * ((mt_rank + it) % 3));
  }
  if (mode == TEST) {
    int flag = 0;
    while (!flag) {
      MT_MPI(MPI_Test(req, &flag, MPI_STATUS_IGNORE));
      if (!flag) {
        usleep(100);
      }
    }
  } else {
    MT_MPI(MPI_Wait(req, MPI_STATUS_IGNORE));
  }
  MT_CHECK(*req == MPI_REQUEST_NULL, "iteration %ld %s: request not freed",
           it, what);
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "nonblocking_collectives");
  int buf[N];
  long lsend[N], lrecv[N];
  for (it = 0; mt_continue(it); it++) {
    int mode = it % NMODES;
    long round = it / NMODES;
    int bcast_root = (int)((it + round) % mt_size);
    int reduce_root = (int)((it + 2 * round + 1) % mt_size);
    if (mt_rank == round % mt_size) {
      usleep(2000);
    }
    for (int i = 0; i < N; i++) {
      buf[i] = mt_rank == bcast_root ? mt_value(bcast_root, it, i) : -1;
      lsend[i] = mt_value(mt_rank, it, N + i);
      lrecv[i] = -1;
    }
    if (mode == WAITALL) {
      MPI_Request reqs[3];
      MT_MPI(MPI_Ibarrier(MPI_COMM_WORLD, &reqs[0]));
      MT_MPI(MPI_Ibcast(buf, N, MPI_INT, bcast_root, MPI_COMM_WORLD,
                        &reqs[1]));
      MT_MPI(MPI_Ireduce(lsend, lrecv, N, MPI_LONG, MPI_SUM, reduce_root,
                         MPI_COMM_WORLD, &reqs[2]));
      MT_MPI(MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE));
      for (int k = 0; k < 3; k++) {
        MT_CHECK(reqs[k] == MPI_REQUEST_NULL,
                 "iteration %ld: request %d not freed", it, k);
      }
    } else {
      MPI_Request req;
      MT_MPI(MPI_Ibarrier(MPI_COMM_WORLD, &req));
      complete(&req, mode, "ibarrier");
      MT_MPI(MPI_Ibcast(buf, N, MPI_INT, bcast_root, MPI_COMM_WORLD, &req));
      complete(&req, mode, "ibcast");
      MT_MPI(MPI_Ireduce(lsend, lrecv, N, MPI_LONG, MPI_SUM, reduce_root,
                         MPI_COMM_WORLD, &req));
      complete(&req, mode, "ireduce");
    }
    for (int i = 0; i < N; i++) {
      MT_CHECK(buf[i] == mt_value(bcast_root, it, i),
               "iteration %ld mode %d ibcast word %d: %d", it, mode, i,
               buf[i]);
    }
    for (int i = 0; mt_rank == reduce_root && i < N; i++) {
      long want = 0;
      for (int j = 0; j < mt_size; j++) {
        want += mt_value(j, it, N + i);
      }
      MT_CHECK(lrecv[i] == want, "iteration %ld mode %d ireduce word %d: "
               "%ld, not %ld", it, mode, i, lrecv[i], want);
    }
  }
  mt_finish(it);
  return 0;
}
