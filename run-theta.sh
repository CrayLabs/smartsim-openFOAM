#!/bin/bash
#COBALT -t 03:00:00
#COBALT -n 128
#COBALT -q default
#COBALT -A datascience

launcher=cobalt   # launcher for the run (slurm or cobalt)
db_nodes=1        # the number of database nodes to use
db_port=6379      # the port to use for database communication
gen_nodes=2       # the number of compute nodes to use for each OpenFOAM data generation case
gen_ppn=64        # the number of processors per node for each OpenFOAM data generationc case
sim_nodes=2       # the number of compute nodes to use for the OpenFOAM inference case
sim_ppn=64        # the number of processors per node for each OpenFOAM inference case

module unload atp
module load miniconda-3/2021-07-28
conda activate /path/to/miniconda3/envs/openfoam/
export SMARTSIM_LOG_LEVEL=developer

export OF_PATH=/path/to/OpenFOAM-5.x
export SMARTREDIS_LIB_PATH=/path/to/SmartRedis/install/lib

cd $OF_PATH
source etc/bashrc
cd -

export LD_LIBRARY_PATH=$SMARTREDIS_LIB_PATH:$FOAM_USER_LIBBIN:$FOAM_LIBBIN:$LD_LIBRARY_PATH

python driver.py --launcher=$launcher --db_nodes=$db_nodes --db_port=$db_port \
                 --gen_nodes=$gen_nodes --gen_ppn=$gen_ppn --sim_nodes=$sim_nodes --sim_ppn=$sim_ppn