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

// The runtime of the libcuda front, whose functions all abort for now: the
// application's driver calls are not supported yet (Phase 2).

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "front.h"

void
front_fatal(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "MANA: --cuda: ");
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  abort();
}

void
front_unsupported(const char *name)
{
  front_fatal("%s: the CUDA driver API is not supported yet; use the CUDA "
              "runtime (cuda/DESIGN-lower-half-cuda.md)\n", name);
}
