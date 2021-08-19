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

The

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

Cray MPI options need to be added to ``OpenFOAM-5.x/etc/config.sh/mpi``. These options are shown in the block below, and ``smartsim-openFOAM/builds/cray_xc/mpi`` shows the proper placement of these options.

```
# Add these Cray compiler settings to OpenFOAM-5.x/etc/config.sh/mpi
MPICH2)
    export FOAM_MPI=mpich2
    export MPI_ARCH_PATH=$MPICH_DIR
    ;;
```

### Adding SmartRedis to OpenFOAM