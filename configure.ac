dnl Use 'autoreconf' to generate configure
AC_INIT([MANA],
        m4_esyscmd([tr -d '\n' < VERSION]),
        [https://github.com/mpickpt/mana/issues],
        [mana],
        [https://github.com/mpickpt/mana])

AC_PREREQ([2.60])
# TODO(kapil): Add 'subdir-objects after automake 1.16 has been released.
# AM_INIT_AUTOMAKE([foreign])
# AM_MAINTAINER_MODE
AC_CANONICAL_HOST
AC_PROG_INSTALL
# Automake uses 'ar cru' by default.  The 'ar' program then issues a warning.
AC_SUBST([ARFLAGS], [cr])
AC_PROG_RANLIB
AC_PROG_MAKE_SET
AC_PROG_LN_S
AC_PROG_MKDIR_P
dnl Due to a strange misfeature in autotools, AC_PROG_CXX will succeed if there
dnl is no CXX compiler and if AC_PROG_CXX is invoked _after_ AC_PROG_CC.
dnl So, we are invoking AC_PROG_CXX _before_ AC_PROG_CC, to take advantage
dnl of an undocumented autotools feature.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_CPP
AC_CONFIG_MACRO_DIR([m4])
AX_CXX_COMPILE_STDCXX([20], [noext], [mandatory])

AX_CHECK_COMPILE_FLAG([-std=gnu11],
                      [CFLAGS="${CFLAGS} -std=gnu11"],
                      [echo "C compiler cannot compile gnu11 code"; exit -1])

dnl gfortran-10 errors out if argument mismatch
dnl See if Fotran compiler supports -fallow-argument-mismatch for legacy code
AC_LANG_PUSH([Fortran])
AX_CHECK_COMPILE_FLAG([-fallow-argument-mismatch],
                      [FFLAGS="${FFLAGS} -fallow-argument-mismatch"],
                      [])
AC_LANG_POP([Fortran])

AC_CHECK_FILE([dmtcp/configure],
  [AC_MSG_RESULT([yes, dmtcp/configure exists])
   AC_DEFINE([HAVE_DMTCP_CONFIGURE], 1, [Define if dmtcp/configure exists])
  ],
  [AC_MSG_ERROR([no, dmtcp/configure does not exist;
                      Did you do:  git submodule init && git submodule update])
  ])

dnl DMTCP's configure refuses the options that it doesn't know: MANA's own.
mana_dmtcp_args=
eval "set x $ac_configure_args"
shift
for mana_arg
do
  case $mana_arg in
    --with-mpi|--with-mpi=*|--without-mpi|--with-cuda|--with-cuda=*|--without-cuda) ;;
    *) mana_dmtcp_args="$mana_dmtcp_args '`AS_ECHO(["$mana_arg"]) | sed "s/'/'\\\\\\\\''/g"`'" ;;
  esac
done
ac_configure_args="$mana_dmtcp_args --disable-dlsym-wrapper"
AC_CONFIG_SUBDIRS([dmtcp])

dnl MANA's layers (cuda/PROPOSAL-code-organization.md).  The core, libmana.so
dnl and bin/lower-half-nompi, is always built.  The MPI layer (libmana_mpi.so,
dnl bin/lower-half) and the CUDA front (mana_launch --cuda) are built if
dnl what they need is found, or as --with-mpi, --with-cuda and their
dnl --without- forms say.  The summary at the end says what and why.

dnl NERSC's Perlmutter, as mpi-proxy-split/Makefile_config.in tells it.
case "$HOST" in
  *nid0*|*login*) mana_perlmutter=yes ;;
  *) mana_perlmutter=no ;;
esac

AC_ARG_WITH([mpi],
  [AS_HELP_STRING([--with-mpi@<:@=DIR@:>@],
    [build the MPI layer, with the MPI compilers in DIR/bin (default: if
     mpicc and mpic++ are found, or cc and CC on Perlmutter)])],
  [], [with_mpi=check])
AC_ARG_VAR([MPICC], [the MPI C compiler (default: mpicc)])
AC_ARG_VAR([MPICXX], [the MPI C++ compiler (default: mpic++)])
AC_ARG_VAR([MPIFORTRAN], [the MPI Fortran compiler (default: mpifort)])
AC_ARG_VAR([MPIRUN], [the MPI launcher (default: mpirun)])
mana_mpi=no
mana_mpi_why="--without-mpi"
if test "$with_mpi" != no; then
  if test "$with_mpi" != yes && test "$with_mpi" != check; then
    mana_mpi_bin="$with_mpi/bin/"
  else
    mana_mpi_bin=
  fi
  if test "$mana_perlmutter" = yes; then
    : ${MPICC:=cc} ${MPICXX:=CC} ${MPIFORTRAN:=ftn} ${MPIRUN:=srun}
  else
    : ${MPICC:=${mana_mpi_bin}mpicc} ${MPICXX:=${mana_mpi_bin}mpic++}
    : ${MPIFORTRAN:=${mana_mpi_bin}mpifort} ${MPIRUN:=${mana_mpi_bin}mpirun}
  fi
  AC_MSG_CHECKING([whether $MPICC and $MPICXX build MPI programs])
  cat > conftest-mpi.c <<_ACEOF
#include <mpi.h>
int main(int argc, char **argv) { return MPI_Init(&argc, &argv); }
_ACEOF
  if $MPICC -o conftest-mpi conftest-mpi.c >&AS_MESSAGE_LOG_FD 2>&1 &&
     $MPICXX -std=c++20 -x c++ -o conftest-mpi conftest-mpi.c \
       >&AS_MESSAGE_LOG_FD 2>&1; then
    mana_mpi=yes
    mana_mpi_why="with $MPICC and $MPICXX"
  else
    mana_mpi_why="$MPICC or $MPICXX cannot build an MPI program (see config.log); give --with-mpi=DIR, or MPICC and MPICXX"
  fi
  rm -f conftest-mpi conftest-mpi.c
  AC_MSG_RESULT([$mana_mpi])
  if test "$mana_mpi" = no && test "$with_mpi" != check; then
    AC_MSG_ERROR([--with-mpi: $mana_mpi_why])
  fi
fi
AC_SUBST([MANA_MPI], [$mana_mpi])

AC_ARG_WITH([cuda],
  [AS_HELP_STRING([--with-cuda@<:@=DIR@:>@],
    [build the CUDA front for mana_launch --cuda, with the CUDA toolkit in
     DIR (default: if nvcc and the CUDA driver are found)])],
  [], [with_cuda=check])
mana_cuda=no
mana_cuda_why="--without-cuda"
CUDA_HOME=
if test "$with_cuda" != no; then
  if test "$with_cuda" != yes && test "$with_cuda" != check; then
    CUDA_HOME="$with_cuda"
  else
    mana_nvcc=`command -v nvcc 2>/dev/null`
    test -z "$mana_nvcc" && test -x /usr/local/cuda/bin/nvcc &&
      mana_nvcc=/usr/local/cuda/bin/nvcc
    test -n "$mana_nvcc" && CUDA_HOME=`dirname "$mana_nvcc"`/..
    test -n "$CUDA_HOME" && CUDA_HOME=`cd "$CUDA_HOME" && pwd`
  fi
  AC_MSG_CHECKING([for the CUDA toolkit and driver])
  mana_libcuda=`/sbin/ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1 / {print $NF; exit}'`
  if test -z "$CUDA_HOME"; then
    mana_cuda_why="no nvcc found; give --with-cuda=DIR"
  elif test ! -x "$CUDA_HOME/bin/nvcc" || test ! -r "$CUDA_HOME/include/cuda.h"; then
    mana_cuda_why="no nvcc or cuda.h in $CUDA_HOME"
  elif test -z "`ls $CUDA_HOME/lib64/libcudart.so.[[0-9]]* 2>/dev/null`"; then
    mana_cuda_why="no libcudart.so in $CUDA_HOME/lib64"
  elif test -z "$mana_libcuda"; then
    mana_cuda_why="no CUDA driver (libcuda.so.1) found by ldconfig"
  elif ! command -v python3 >/dev/null 2>&1; then
    mana_cuda_why="no python3, which generates the front"
  else
    mana_cuda=yes
    mana_cuda_why="`$CUDA_HOME/bin/nvcc --version | sed -n 's/.*release \([[0-9.]]*\).*/CUDA \1/p'` in $CUDA_HOME"
  fi
  AC_MSG_RESULT([$mana_cuda])
  if test "$mana_cuda" = no && test "$with_cuda" != check; then
    AC_MSG_ERROR([--with-cuda: $mana_cuda_why])
  fi
fi
dnl The libraries that the front covers too (cuBLAS, cuFFT, cuRAND): next to
dnl libcudart, or, in NVIDIA's HPC SDK (cudatoolkit at NERSC), in
dnl math_libs/VERSION/lib64 beside cuda/VERSION.
CUDA_MATH_LIBDIR=
if test "$mana_cuda" = yes; then
  AC_MSG_CHECKING([for the CUDA math libraries])
  for mana_dir in "$CUDA_HOME/lib64" \
      "$CUDA_HOME/../../math_libs/`basename "$CUDA_HOME"`/lib64" \
      "$CUDA_HOME/../../math_libs/lib64"; do
    if test -n "`ls $mana_dir/libcublas.so.[[0-9]]* 2>/dev/null`"; then
      CUDA_MATH_LIBDIR=`cd "$mana_dir" && pwd`
      break
    fi
  done
  if test -n "$CUDA_MATH_LIBDIR"; then
    mana_cuda_why="$mana_cuda_why; cuBLAS, cuFFT, cuRAND in $CUDA_MATH_LIBDIR"
  else
    mana_cuda_why="$mana_cuda_why; no cuBLAS, cuFFT, cuRAND"
  fi
  AC_MSG_RESULT([${CUDA_MATH_LIBDIR:-no}])
fi
AC_SUBST([MANA_CUDA], [$mana_cuda])
AC_SUBST([CUDA_HOME])
AC_SUBST([CUDA_MATH_LIBDIR])

AC_CONFIG_FILES([Makefile \
                 manpages/Makefile \
                 mpi-proxy-split/Makefile_config])

dnl Autoconf manual says option checking is set to warn ("yes") by
dnl  by default.  But it's actually set to "no".
dnl So, we enforce our own choice ("fatal") if autoconf won't cooperate.
enable_option_checking=fatal
if test -n "$ac_unrecognized_opts"; then
  case $enable_option_checking in
    no) ;;
    fatal) { AS_ECHO(["$as_me: error: unrecognized options: $ac_unrecognized_opts"]) >&2
   { (exit 1); exit 1; }; } ;;
    *)     AS_ECHO(["$as_me: WARNING: unrecognized options: $ac_unrecognized_opts"]) >&2 ;;
  esac
fi

AC_ARG_ENABLE([debug],
            [AS_HELP_STRING([--enable-debug],
                            [Use debugging flags "-Wall -g3 -O0" on DMTCP libs
                             (default is disabled)])],
            [use_debug=$enableval],
            [use_debug=no])

if test "$use_debug" = "yes"; then
  AC_SUBST([DEBUG], [yes])
  AC_DEFINE([DEBUG],[1],[Use debugging flags "-Wall -g3 -O0"])
  CFLAGS="$CFLAGS -Wall -g3 -O0 -DDEBUG"
  CPPFLAGS="$CPPFLAGS -Wall -g3 -O0 -DDEBUG"
  CXXFLAGS="$CXXFLAGS -Wall -g3 -O0 -DDEBUG"
else
  AC_SUBST([DEBUG], [no])
fi

AC_SUBST([FFLAGS], [$FFLAGS])

# Keeping these DMTCP flags only for convenience.
AC_ARG_ENABLE([quiet])
AC_ARG_ENABLE([fsgsbase-override])

#check for pandoc
AC_PATH_PROG(PANDOC, [pandoc], [no], [/usr/bin:/bin])

AC_OUTPUT

cat <<_ACEOF

MANA will build:
  the core (libmana.so, bin/lower-half-nompi):  yes
  the MPI layer (libmana_mpi.so, bin/lower-half):  $mana_mpi: $mana_mpi_why
  the CUDA front (mana_launch --cuda):  $mana_cuda: $mana_cuda_why
  the NCCL front (mana_launch --nccl):  no: not written yet
_ACEOF
