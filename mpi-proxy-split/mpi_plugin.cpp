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

// MANA's MPI layer, libmana_mpi.so, with the MPI wrappers (mpi-wrappers/),
// virtual IDs, the drains and the replay.  libmana.so, the core
// (mana_core.cpp), calls its steps at each checkpoint and restart.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "mpi_files.h"
#include "mpi_plugin.h"
#include "mana_core.h"
#include "lower-half-api.h"
#include "lower_half_ckpt.h"
#include "p2p_log_replay.h"
#include "p2p_drain_send_recv.h"
#include "seq_num.h"
#include "mpi_nextfunc.h"
#include "virtual_id.h"
#include "logging.h"

#include "dmtcp.h"
#include "dmtcp_assert.h"
#include "procselfmaps.h"

extern "C" int MPI_MANA_Internal(char *dummy);

using namespace dmtcp;

extern std::unordered_map<MPI_File, OpenFileParameters> g_params_map;

ProcSelfMaps *preMpiInitMaps = nullptr;
ProcSelfMaps *postMpiInitMaps = nullptr;
map<void*, size_t> *mpiInitLhAreas = nullptr;

// Check if haystack region contains needle region.
static inline int
regionContains(const void *haystackStart,
               const void *haystackEnd,
               const void *needleStart,
               const void *needleEnd)
{
  return needleStart >= haystackStart && needleEnd <= haystackEnd;
}

// These are called from the MPI_Init wrapper in mpi-wrappers/mpi_wrappers.cpp
// FIXME: But during restart, MPI_Init is called from getRankFptr (aka getRank),
//        which calls MPI_Init before MPI_Comm_rank.  If this is a
//        restarted process (not a 'launch' process), then these functions
//        are not called, and we miss recording the ioctl of /anon_hugepage .
//          To fix this, restart_plugin needs to call this after getRankFptr,
//        and simply record all memory regions as part of the lower half.
//        Note that restart_plugin is part of mtcp_restart (no libc available).
void recordPreMpiInitMaps()
{
  if (preMpiInitMaps != nullptr) {
    return;
  }
  mpiInitLhAreas = new map<void*, size_t>();
  preMpiInitMaps = new ProcSelfMaps();
}

void recordPostMpiInitMaps()
{
  if (postMpiInitMaps != nullptr) {
    return;
  }
  postMpiInitMaps = new ProcSelfMaps();
  ProcMapsArea area;

  {
    ostringstream o;
    const void *procSelfMapsData = postMpiInitMaps->getData();

    while (postMpiInitMaps->getNextArea(&area)) {
      if (!regionContains(area.addr, area.endAddr,
                          procSelfMapsData, procSelfMapsData)) {
        mpiInitLhAreas->insert(std::make_pair(area.addr, area.size));;
      }
    }

    // Now remove those mappings that existed before Mpi_Init.
    ASSERT_NOT_NULL(preMpiInitMaps);
    while (preMpiInitMaps->getNextArea(&area)) {
      if (mpiInitLhAreas->find(area.addr) != mpiInitLhAreas->end()) {
        WARN(mpiInitLhAreas->at(area.addr) == area.size, "addr={} size={}",
             (void *)area.addr, area.size);
        mpiInitLhAreas->erase(area.addr);
      } else {
        // Check if a region has the same end addr. E.g., thread stack grew.
        for (auto it : *mpiInitLhAreas) {
          void *endAddr = (char*) it.first + it.second;
          if (endAddr == area.endAddr) {
            mpiInitLhAreas->erase(area.addr);
            break;
          }
        }
      }
    }
  }
}

void recordMpiInitMaps()
{
}

const char *
get_mpi_file_filename()
{
  struct stat st;
  const char *ckptDir;
  dmtcp::ostringstream o;

  ckptDir = dmtcp_get_ckpt_dir();
  if (stat(ckptDir, &st) == -1) {
    mkdir(ckptDir, 0700); // Create directory if not already exist
  }
  o << ckptDir << "/mpi_files.mana";
  return strdup(o.str().c_str());
}

static void
save_mpi_files(const char *filename)
{
  // No need to save open files if there are no files to save
  if (g_params_map.size() == 0) return;

  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0755);
  if (fd == -1) {
    return;
  }

  for (auto itr = g_params_map.begin(); itr != g_params_map.end(); itr++) {
    // Only the most recent value of these properties needs to be saved
    MPI_File_get_position(itr->first, &itr->second._offset);
    MPI_File_get_atomicity(itr->first, &itr->second._atomicity);
    MPI_File_get_size(itr->first, &itr->second._size);

    // Note that a garbage view may be set by this call if set_view was never
    // called, but that view won't be restored
    MPI_File_get_view(itr->first, &itr->second._viewDisp,
                      &itr->second._viewEType, &itr->second._viewFType,
                      itr->second._datarep);

    // Save both virtual ID and corresponding parameters
    write(fd, &itr->first, sizeof(MPI_File));
    write(fd, &itr->second, sizeof(struct OpenFileParameters));
  }

  close(fd);
}

static void
restore_mpi_files(const char *filename)
{
  int fd = open(filename, O_RDONLY, 0755);

  // It's fine if the file doesn't exist. This just means there's no files
  // to be restored.
  if (fd == -1) {
    return;
  }

  // Restore the g_params_map with each virtual file and its corresponding
  // parameters from the mapping file
  MPI_File file_buf;
  struct OpenFileParameters params_buf;
  g_params_map.clear();
  size_t bytes_read = 0;
  do {
    bytes_read = read(fd, &file_buf, sizeof(MPI_File));
    bytes_read = read(fd, &params_buf, sizeof(OpenFileParameters));
    g_params_map[file_buf] = params_buf;
  } while(bytes_read != 0);
  close(fd);

  // Restore MPI-created files
  for (auto itr = g_params_map.begin(); itr != g_params_map.end(); itr++) {
    // Re-create file with initial parameters
    int retval;
    MPI_File fh;
    MPI_Comm real_comm = get_real_id((mana_mpi_handle){.comm = itr->second._comm}).comm;
    MPI_Info real_info = lower_half_handle(itr->second._info);
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    retval = NEXT_FUNC(File_open)(real_comm, itr->second._filepath,
                                  itr->second._mode, real_info, &fh);
    RETURN_TO_UPPER_HALF();
    ASSERT(retval == 0, "Restoration of MPI_File_open failed");

    // Update virtual mapping with newly created file
    update_virt_id((mana_mpi_handle){.file = itr->first}, (mana_mpi_handle){.file = fh});

    // Restore file characteristics that are universal to the entire file
    // These are characteristics that do not have to be replayed in order,
    // because only the most recent one will affect the behavior of the file

    // Note that the order of restoration is extremely important (the file size)
    // MUST be set before the view or offset
    MPI_File_set_atomicity(fh, itr->second._atomicity);
    MPI_File_set_size(fh, itr->second._size);

    // Only restore view if required
    if (itr->second._viewSet) {
      MPI_File_set_view(fh, itr->second._viewDisp, itr->second._viewEType,
                        itr->second._viewFType, itr->second._datarep,
                        itr->second._viewInfo);
    }
    MPI_File_seek(fh, itr->second._offset, MPI_SEEK_SET);

  }

}

// No thread may be in the lower half when DMTCP suspends the threads.
static void
close_lower_half()
{
  uint64_t t = drainStatsNow();
  wait_for_threads_to_leave_lower_half();  // lower_half_ckpt.cpp
  g_drain_stats.t_wait_lower_half += drainStatsNow() - t;
}

static void
close_lower_half_except_blocked()
{
  uint64_t t = drainStatsNow();
  wait_for_unblocked_threads_to_leave_lower_half();  // lower_half_ckpt.cpp
  g_drain_stats.t_wait_lower_half += drainStatsNow() - t;
}

// When this checkpoint's agreement began, for the drain's statistics.
static uint64_t g_drain_t0;

// The MPI layer's steps (mana_core.h), on the checkpoint thread.
static void
mpi_layer_step(int step)
{
  switch (step) {
    case MANA_STEP_WAIT_FOR_INIT: {
      // A rank still in MPI_Init has no MANA state yet (no rank, no
      // g_world_comm).  Its MPI_Init can need the other ranks' progress,
      // which their application threads keep making until the drains.
      while (!__atomic_load_n(&g_libmpi_is_initialized, __ATOMIC_ACQUIRE)) {
        usleep(1000);
      }
      break;
    }

    case MANA_STEP_AGREE: {
      // In each pass of the layers' agreements (mana_core.h): the Collective
      // Clock.  A later pass starts from the last one's targets.
      if (mana_drain_pass() == 0) {
        resetDrainStats();  // p2p_drain_send_recv.cpp
        g_drain_t0 = drainStatsNow();
      }
      mana_state = CKPT_COLLECTIVE;
      // preSuspendBarrier() will send coord response and get worker state.
      // FIXME:  See commant at: dmtcpplugin.cpp:'case DMTCP_EVENT_PRESUSPEND'
      drain_mpi_collective();
      break;
    }

    case MANA_STEP_DRAIN: {
      uint64_t t0 = g_drain_t0;
      // From here on, the checkpoint thread calls MPI.  With
      // MANA_P2P_WAIT=polling, no thread waits in the lower half for the
      // drain, so close it now (the lower half runs as MPI_THREAD_SINGLE).  In
      // blocking mode, a blocked MPI_Recv stays there until the P2P drain
      // sends it a dummy, and the other threads leave now: only the
      // checkpoint thread may complete an MPI_Issend that has a notice
      // (publishSsendNotices()).  If the lower half runs as
      // MPI_THREAD_MULTIPLE, the checkpoint thread uses a lower-half TLS of
      // its own.
      bool own_tls = lh_info->ckpt_fsaddr != NULL;
      if (g_p2p_wait == P2P_WAIT_POLLING) {
        close_lower_half();
      } else {
        close_lower_half_except_blocked();
      }
      dmtcp_global_barrier("MPI:Drain-Send-Recv");
      g_drain_stats.t_collective =
        drainStatsNow() - t0 - g_drain_stats.t_wait_lower_half;
      mana_state = CKPT_P2P;
      if (own_tls) {
        lh_info->ckpt_uh_fs = (void *)getFS();
      }
      drainP2p(); // p2p_drain_send_recv.cpp
      lh_info->ckpt_uh_fs = NULL;
      if (g_p2p_wait == P2P_WAIT_BLOCKING) {
        close_lower_half();
      }
      // The lower half is closed now in both modes.  Without barriers
      // (drainHasNoBarrier()), drainP2p() has completed the non-blocking
      // collectives.
      if (!drainHasNoBarrier()) {
        uint64_t t_nbc = drainStatsNow();
        complete_pending_nonblocking_collectives();  // seq_num.cpp
        g_drain_stats.t_collective += drainStatsNow() - t_nbc;
      }
      reportDrainStats();  // With MANA_DRAIN_STATS set
      break;
    }

    case MANA_STEP_PRECHECKPOINT: {
      recordMpiInitMaps();
      dmtcp_local_barrier("MPI:GetLocalRankInfo");
      getLocalRankInfo(); // p2p_log_replay.cpp
      mana_set_rank(g_world_rank);
      mana_set_header("p2p_wait", g_p2p_wait == P2P_WAIT_BLOCKING ?
                                  "blocking" : "polling");
      break;
    }

    case MANA_STEP_SAVE: {
      const char *file = get_mpi_file_filename();
      save_mpi_files(file);
      break;
    }

    case MANA_STEP_RESUME: {
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      seq_num_reset();
      dmtcp_local_barrier("MPI:seq_num_reset");
      sendOwedSsendAcks(); // p2p_drain_send_recv.cpp
      break;
    }

    case MANA_STEP_RESTART: {
      mpiInitLhAreas->clear();
      dmtcp_local_barrier("MPI:updateEnviron");
      seq_num_reset();
      dmtcp_local_barrier("MPI:seq_num_reset");
      init_predefined_virt_ids();
      check_tag_ub();
      reconstruct_descriptors();
      dmtcp_local_barrier("MPI:Reset-Drain-Send-Recv-Counters");
      resetDrainCounters(); // p2p_drain_send_recv.cpp
      char str[1];
      MPI_MANA_Internal(str); // This does nothing.  Modify in lower-half
                             // for easy debugging of lower half during restart.
                             // See definition in mpi-wrappers/mpi_wrappers.cpp
      break;
    }

    case MANA_STEP_REPLAY: {
      mana_state = RESTART_REPLAY;
      dmtcp_global_barrier("MPI:replayMpiP2pOnRestart");
      replayMpiP2pOnRestart(); // p2p_log_replay.cpp
      sendOwedSsendAcks(); // p2p_drain_send_recv.cpp
      dmtcp_local_barrier("MPI:p2p_log_replay.cpp-void");
      const char *file = get_mpi_file_filename();
      restore_mpi_files(file);
      dmtcp_local_barrier("MPI:Restore-MPI-Files");
      break;
    }

    case MANA_STEP_EXIT: {
      seq_num_destroy();
      break;
    }

    default: {
      break;
    }
  }
}

// libmana_mpi.so is loaded at launch, after libmana.so (its dependency).
__attribute__((constructor)) static void
mpi_layer_register()
{
  seq_num_init();
  mana_register_layer(MANA_LAYER_MPI, mpi_layer_step);
}
