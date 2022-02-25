# Building on a Cray XC

The following instructions detail how to build OpenFOAM-5.x,
ThirdParty-5.x, and other dependencies on a Cray XC.

## Environment

While SmartSim and SmartRedis are compatible with multiple compilers,
OpenFOAM-5.x compile errors have been encountered with clang compilers.
As a result, it is recommended that GNU compilers be used.  If available,
the ``PrgEnv-gnu`` module should be swapped with the default ``PrgEnv``.
If the ``PrgEnv-cray`` module is loaded by default on your Cray XC system,
execute:

``` bash
module swap PrgEnv-cray PrgEnv-gnu
```

On a the ANL Theta system, users should execute the following to set up
a build environment for OpenFOAM-5.x and ThirdParty-5.x:

```bash
module purge
module load PrgEnv-gnu
module load cray-mpich
module unload atp perftools-base cray-libsci
export CRAYPE_LINK_TYPE=dynamic
```

## Building OpenFOAM-5.x

These directions were adapted from public instructions for building OpenFOAM
on the Titan supercomputer and should be applicable to a wide
range of Cray systems.  However, these instructions were tested
specifically on a Cray XC system.

Clone the OpenFOAM-5.x, OpenFOAM ThirdParty-5.x, and smartsim-openFOAM repositories in
the same top-level directory:

```
git clone https://github.com/OpenFOAM/OpenFOAM-5.x.git
git clone https://github.com/OpenFOAM/ThirdParty-5.x.git
git clone https://github.com/CrayLabs/smartsim-openFOAM
```

Copy the Cray XC build rules provided in ``smartsim-openFOAM`` to the
``OpenFOAM-5.x`` build rules directory:

```
mkdir OpenFOAM-5.x/wmake/rules/crayxcGcc
```
```
cp -r smartsim-openFOAM/builds/cray_xc/rules/* OpenFOAM-5.x/wmake/rules/crayxcGcc/
```

Copy the OpenFOAM ``bashrc`` file that contains edits for
``WM_ARCH_OPTION=cray`` and ``WM_MPLIB=MPICH2``:

```bash
cp smartsim-openFOAM/builds/cray_xc/bashrc OpenFOAM-5.x/etc/bashrc
```

Now, copy the custom compiler options:

```bash
cp smartsim-openFOAM/builds/cray_xc/settings OpenFOAM-5.x/etc/config.sh/settings
```

Now, copy the custom MPI settings.  Verify that your current environment has ``MPICH_DIR`` set to the correct path.  If not, this ``MPICH_DIR`` should be set to the MPICH install directory.

```bash
cp smartsim-openFOAM/builds/cray_xc/mpi OpenFOAM-5.x/etc/config.sh/mpi
```

The OpenFOAM build environment can now be set up with:

```
cd OpenFOAM-5.x && source etc/bashrc && cd ..
```

Before OpenFOAM can be built, third-party dependencies need to be built.  To build these dependencies on the Cray XC, copy the build settings provided in ``smartsim-openFOAM`` to ``ThirdParty-5.x``:

```
cp smartsim-openFOAM/builds/cray_xc/Makefile.inc.i686_pc_linux2.shlib-OpenFOAM ThirdParty-5.x/etc/wmakeFiles/scotch/
```

Now, build the OpenFOAM third-party dependencies:

```
cd ThirdParty-5.x && ./Allwmake -j 8 && cd ..
```

OpenFOAM can now be built:

```
cd OpenFOAM-5.x && ./Allwmake -j 8 && cd ..
```