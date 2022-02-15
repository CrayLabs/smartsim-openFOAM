#!/bin/bash
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=SS-OpenFOAM
#SBATCH --output=SS-OpenFOAM.out
#SBATCH --error=SS-OpenFOAM.err

db_nodes=1        # the number of database nodes to use
db_port=6379      # the port to use for database communication
gen_nodes=2       # the number of compute nodes to use for each OpenFOAM data generation case
gen_ppn=24        # the number of processors per node for each OpenFOAM data generationc case
sim_nodes=2       # the number of compute nodes to use for the OpenFOAM inference case
sim_ppn=24        # the number of processors per node for each OpenFOAM inference case

module unload atp
export SMARTSIM_LOG_LEVEL=developer

export OF_PATH=/path/to/OpenFOAM-5.x
export SMARTREDIS_LIB_PATH=/path/to/SmartRedis/install/lib

cd $OF_PATH
source etc/bashrc
cd -

export LD_LIBRARY_PATH=$SMARTREDIS_LIB_PATH:$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$LD_LIBRARY_PATH

python driver.py --db_nodes=$db_nodes --db_port=$db_port \
                 --gen_nodes=$gen_nodes --gen_ppn=$gen_ppn --sim_nodes=$sim_nodes --sim_ppn=$sim_ppn
