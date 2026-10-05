#include <mpi.h>
#include <stdio.h>
#include <string.h>
int main(int c, char **v) {
  MPI_Init(&c, &v);
  int r; MPI_Comm_rank(MPI_COMM_WORLD, &r);
  if (r == 0) {
    FILE *f = fopen("/proc/self/maps", "r"); char l[512];
    while (fgets(l, sizeof l, f)) if (strncmp(l, "200000000", 9) == 0 || strstr(l, "nvidia0")) fputs(l, stdout);
  }
  MPI_Finalize();
}
