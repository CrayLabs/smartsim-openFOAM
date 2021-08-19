# smartsim-openFOAM

This repository provides instructions, source files, and
examples for including the SmartRedis client in
OpenFOAM so that OpenFOAM can leverage the machine learning,
data analysis, and data visualization capabilities in SmartSim.

## Building OpenFOAM 5.x with SmartRedis

This section outlines the steps required to build OpenFOAM 5.x
with SmartRedis.  OpenFOAM provides
[installation instructions](https://develop.openfoam.com/Development/openfoam/-/blob/master/doc/Build.md)
for various environments.  Specific instructions
are provided herein for building OpenFOAM from source
on a Cray XC.  If a Cray XC is not being used,
skip ahead to [Adding SmartRedis to OpenFoam](#markdown-Adding-SmartRedis-to-OpenFOAM) after successfully
installing OpenFOAM 5.x on your system.

### Building OpenFOAM on a Cray XC

These directions were adapted from public instructions for building OpenFOAM on the Titan supercomputer and should be applicable to a wide range of Cray systems.  However, these instructions were tested specifically on a Cray XC system.

If the ``PrgEnv-cray`` module is loaded by default on your Cray XC system, execute:

``` bash
module swap PrgEnv-cray PrgEnv-gnu
```

Now, clone the OpenFOAM-5.x, OpenFOAM ThirdParty-5.x, and smartsim-openFOAM repositories in
the same top-level directory:

```
git clone https://github.com/OpenFOAM/OpenFOAM-5.x.git
git clone https://github.com/OpenFOAM/ThirdParty-5.x.git
git clone https://github.com/CrayLabs/smartsim-openFOAM
```

Copy the Cray XC build rules provided in smartsim-openFOAM to the
OpenFOAM-5.x build rules directory:

```
mkdir OpenFOAM-5.x/wmake/rules/crayxcGcc
```
```
cp -r smartsim-openFOAM/builds/cray_xc/rules/ OpenFOAM-5.x/wmake/rules/crayxcGcc/
```

Now that there are build rules for the Cray XC system, modify ``OpenFOAM-5.x/etc/bashrc`` to specify ``cray`` for ``WM_ARCH_OPTION``:

```
# Locate WM_ARCH_OPTION in OpenFOAM-5.x/etc/bashrc and set to "cray"
WM_ARCH_OPTION=cray
```

Also, the MPI library rules contained in the XC rules must be specified in the ``OpenFOAM-5.x/etc/bashrc`` file.  Locate ``WM_MPLIB`` and set to ``MPICH2``:

```
# Locate WM_MPLIB in OpenFOAM-5.x/etc/bashrc and set to "MPICH2"
WM_MPLIB=MPICH2
```

Cray specific compiler options need to be added to ``OpenFOAM-5.x/etc/config.sh/settings``. These options are shown in the block below, and ``smartsim-openFOAM/builds/cray_xc/settings`` shows the proper placement of these options.

```
# Add these Cray compiler settings to OpenFOAM-5.x/etc/config.sh/settings
cray)
    WM_ARCH=crayxc
    export WM_COMPILER_LIB_ARCH=64
    export WM_CC='cc'
    export WM_CXX='CC'
    export WM_CFLAGS='-fPIC'
    export WM_CXXFLAGS='-fPIC'
    ;;
```

Cray MPI settings need to be added to ``OpenFOAM-5.x/etc/config.sh/mpi``. These options are shown in the block below, and ``smartsim-openFOAM/builds/cray_xc/mpi`` shows the proper placement of these options.  Verify that your current environment has ``$MPICH_DIR`` set to the correct path.  If not, this ``MPICH_DIR`` should be set to the MPICH install directory.

```
# Add these Cray MPI settings to OpenFOAM-5.x/etc/config.sh/mpi
MPICH2)
    export FOAM_MPI=mpich2
    export MPI_ARCH_PATH=$MPICH_DIR
    ;;
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
cd ThirdParty-5.x && ./Allwmake -j 8 && cd..
```

OpenFOAM can now be built:

```
cd OpenFOAM-5.x && ./Allwmake -j 8 && cd..
```


### Adding SmartRedis to OpenFOAM