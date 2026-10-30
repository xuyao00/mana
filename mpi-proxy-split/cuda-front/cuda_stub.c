/****************************************************************************
 *  Copyright (C) 2026 by Yao Xu                                            *
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
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// The stubs of libcudart and libcuda, which the upper half loads in their
// place in a run without --cuda (../mana_audit.c).  A program built with
// NERSC's cudatoolkit module links libcuda.so.1 even if it makes no CUDA
// call (Cray's cc adds -lcuda), and so runs; a CUDA call ends it.
// gen-cuda-front.py --stub makes each function of the real library a jump
// here, with its name.

#include <stdio.h>
#include <stdlib.h>

__attribute__((visibility("hidden"), noreturn)) void
mana_cuda_stub_called(const char *name)
{
  fprintf(stderr, "MANA: %s: a CUDA call, but MANA was launched without "
          "--cuda (mana_launch --cuda, or MANA_CUDA=1)\n", name);
  abort();
}
