from smartsim.settings import SrunSettings
from smartsim.database import SlurmOrchestrator
from smartsim import Experiment, slurm, constants
from smartredis import Client
from os import environ, getcwd
import time

exp = Experiment(name="openfoam_ml", launcher="slurm")

def get_openfoam_env_vars():
    """Return the environment variables for OpenFOAM

    This function returns the environment variables
    in the current environment that are related to
    OpenFOAM

    :return: dictionary of environment variables
    :rtype: dict with key str and value str
    """

    env_vars = {}
    for key, val in environ.items():
        if len(key)>2 and "WM_" in key[0:3]:
            env_vars[key] = val
        if len(key)>4 and "FOAM_" in key[0:5]:
            env_vars[key] = val
        if key == "MPI_BUFFER_SIZE":
            env_vars[key] = val
        if key == "MPI_ARCH_INC":
            env_vars[key] = val

    return env_vars

def start_database(port, nodes, cpus, tpq):
    """Create and start the Redis database for the scaling test

    This function launches the redis database instances as a
    Sbatch script.

    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :param tpq: number of threads per queue
    :type tpq: int
    :return: orchestrator instance
    :rtype: Orchestrator
    """
    db = SlurmOrchestrator(port=port,
                            db_nodes=nodes,
                            batch=True,
                            threads_per_queue=tpq)
    db.set_cpus(cpus)
    db.set_walltime("1:00:00")
    db.set_batch_arg("exclusive", None)
    exp.generate(db)
    exp.start(db)
    return db

def set_model(model_file, device, batch_size, address, cluster):
    """Set the Tensorflow openFOAM ML model in the orchestrator

    :param model_file: A full path to the model file
    :type model_file: str
    :param device: The device to use for model evaluation
                   (e.g. CPU or GPU)
    :type device: str
    :param batch_size: The batch size to use model evaluation
    :type batch_size: int
    :param address: The address to use for client connection
    :type address: str
    :param cluster: Boolean for cluster or non-cluster connection
    :type cluster: bool
    """

    client = Client(address=address, cluster=cluster)
    time.sleep(10)
    client.set_model_from_file("ml_sa_cg_model",
                                model_file,
                                "TF",
                                device,
                                batch_size,
                                0,
                                "v0.0",
                                ["input_placeholder"],
                                ["output_value/BiasAdd"])

if __name__ == "__main__":

    # Orchestrator settings
    db_node_count = 3
    db_cpus = 36
    db_tpq = 4
    db_port = 6379

    # Simulation settings
    sim_node_count = 1
    sim_dir = "../sim_inputs/pitzDaily_ML/"
    sim_tasks_per_node = 16

    # Model settings
    model_file = "ML_SA_CG.pb"
    device = "CPU"
    batch_size = 1

    # Launch orchestrator
    db = start_database(db_port, db_node_count, db_cpus, db_tpq)

    # Retrieve one of the orchestrator addresses to set
    # the ML model into the database
    address = db.get_address()[0]

    # Set the model into the database
    set_model(model_file, device, batch_size, address, bool(db_node_count>1))

    # Get simulation allocation
    allocation = slurm.get_allocation(nodes=sim_node_count,
                                      time="10:00:00",
                                      options={"exclusive": None,
                                               "job-name": "openfoam"})

    # Retrieve OpenFOAM environment variables for execution
    foam_env_vars = get_openfoam_env_vars()

    # Create the run settings for the mesh decomposition

    decomp_srun = SrunSettings(exe = environ['FOAM_APPBIN'] + "/decomposePar",
                               env_vars = foam_env_vars,
                               alloc = allocation)
    decomp_srun.set_nodes(1)
    decomp_srun.set_tasks(1)

    # Create a SmartSim model that will copy simulation files
    # and then preprocess the mesh for parallel runs
    decomp_model = exp.create_model("openfoam", decomp_srun)

    # Set the model to copy input files
    decomp_model.attach_generator_files(to_copy=[sim_dir])

    # Generate the experiment directory
    exp.generate(decomp_model, overwrite=True)

    # Run the openFOAM parallel decomposition
    exp.start(decomp_model)

    # Definte the simulation directory based on the
    # experiment name and the previous model name used
    exp_dir = getcwd() + '/openfoam_ml/openfoam'

    # Create the run settings for the simulation model
    openfoam_srun = SrunSettings(environ['FOAM_APPBIN'] + "/simpleFoam_ML",
                                 exe_args = "-parallel",
                                 env_vars = foam_env_vars,
                                 alloc = allocation)
    openfoam_srun.set_nodes(sim_node_count)
    openfoam_srun.set_tasks(sim_tasks_per_node)

    # Create the simulation model
    openfoam_model = exp.create_model("openfoam_sim", openfoam_srun, path=exp_dir)

    # Start teh simulation model
    exp.start(openfoam_model)

    # Create the run settings for recombining
    recon_srun = SrunSettings(exe = environ['FOAM_APPBIN'] + "/reconstructPar",
                               env_vars = foam_env_vars,
                               alloc = allocation)
    recon_srun.set_nodes(1)
    recon_srun.set_tasks(1)

    # Create the reconstruction model
    openfoam_recon = exp.create_model("openfoam_recon", recon_srun, path=exp_dir)

    # Start the reconstrucion model
    exp.start(openfoam_recon)




