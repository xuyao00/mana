// External driver of the CUDA checkpoint API: lock, checkpoint, restore,
// unlock the CUDA state of process PID (like NVIDIA's cuda-checkpoint tool).
//   ckpt_ext PID [MODE]   MODE: full (default) | lockonly
// Between checkpoint and restore, prints PID's nvidia mappings and fds.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

static void show(int pid, const char *when) {
  char cmd[256];
  printf("--- %s: nvidia mappings / fds of %d\n", when, pid);
  fflush(stdout);
  snprintf(cmd, sizeof cmd, "grep nvidia /proc/%d/maps | awk '{print $1, $6}'; "
           "ls -l /proc/%d/fd | grep -c nvidia", pid, pid);
  system(cmd);
}

int main(int argc, char **argv) {
  int pid = atoi(argv[1]);
  int lockonly = argc > 2 && strcmp(argv[2], "lockonly") == 0;
  CUprocessState st;
  CUcheckpointLockArgs la = {0}; CUcheckpointCheckpointArgs ca = {0};
  CUcheckpointRestoreArgs ra = {0}; CUcheckpointUnlockArgs ua = {0};
  show(pid, "before");
  printf("lock %d\n", cuCheckpointProcessLock(pid, &la));
  if (!lockonly) {
    printf("ckpt %d\n", cuCheckpointProcessCheckpoint(pid, &ca));
    cuCheckpointProcessGetState(pid, &st); printf("state %d\n", st);
    show(pid, "checkpointed");
    printf("restore %d\n", cuCheckpointProcessRestore(pid, &ra));
  }
  printf("unlock %d\n", cuCheckpointProcessUnlock(pid, &ua));
  cuCheckpointProcessGetState(pid, &st); printf("state %d\n", st);
  show(pid, "after");
  return 0;
}
