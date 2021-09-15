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

def run_training(alloc, training_dir, training_node_count,
                 training_tasks_per_node):
    """Run the TensorFlow training script

    :param alloc: The allocation on which to run
    :type alloc: str
    :param training_dir: The directory where the training
                         script and training data are located
    :type training_dir: str
    :param training_node_count: The number of compute nodes
                                to use for the simulation
    :type training_node_count: int
    :param training_tasks_per_node: The number of tasks
                                    per compute node
    :type training_tasks_per_node: int
    """

    # Create the run settings for the training script
    training_srun = SrunSettings(exe = "python",
                                 exe_args = "ML_Model.py",
                                 alloc = alloc)
    training_srun.set_nodes(training_node_count)
    training_srun.set_tasks(training_tasks_per_node)

    # Create a SmartSim model that will execute the training model
    training_model = exp.create_model("training", training_srun)

    # Set the model to copy input files
    training_model.attach_generator_files(to_copy=[training_dir])

    # Generate the experiment directory
    exp.generate(training_model, overwrite=True)

    # Run the openFOAM parallel decomposition
    exp.start(training_model)

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
                                ["x"],
                                ["Identity"])

def run_decomposition(alloc, foam_env_vars, sim_dir):
    """Run the decomposition step to be be able to run
    OpenFOAM in parallel.

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param sim_dir: The directory where input files are
                    located
    :type sim_dir: str
    """

    # Create the run settings for the mesh decomposition
    decomp_srun = SrunSettings(exe = foam_env_vars['FOAM_APPBIN'] + "/decomposePar",
                               env_vars = foam_env_vars,
                               alloc = alloc)
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

def run_simulation(alloc, foam_env_vars, sim_node_count, sim_tasks_per_node):
    """Run the openFOAM simulation

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param sim_node_count: The number of compute nodes
                           to use for the simulation
    :type sim_node_count: int
    :param sim_tasks_per_node: The number of tasks
                               per compute node
    :type sim_tasks_per_node: int
    """

    # Definte the simulation directory based on the
    # experiment name and the previous model name used
    exp_dir = getcwd() + '/openfoam_ml/openfoam'

    # Create the run settings for the simulation model
    openfoam_srun = SrunSettings(foam_env_vars['FOAM_APPBIN'] + "/simpleFoam_ML",
                                 exe_args = "-parallel",
                                 env_vars = foam_env_vars,
                                 alloc = allocation)
    openfoam_srun.set_nodes(sim_node_count)
    openfoam_srun.set_tasks(sim_tasks_per_node)

    # Create the simulation model
    openfoam_model = exp.create_model("openfoam_sim", openfoam_srun, path=exp_dir)

    # Start teh simulation model
    exp.start(openfoam_model)

def run_reconstruction(alloc, foam_env_vars):
    """Run the openFOAM reconstruction step after a
    parallel run.

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    """

    # Definte the simulation directory based on the
    # experiment name and the previous model name used
    exp_dir = getcwd() + '/openfoam_ml/openfoam'

    # Create the run settings for recombining
    recon_srun = SrunSettings(exe = foam_env_vars['FOAM_APPBIN'] + "/reconstructPar",
                              env_vars = foam_env_vars,
                              alloc = allocation)
    recon_srun.set_nodes(1)
    recon_srun.set_tasks(1)

    # Create the reconstruction model
    openfoam_recon = exp.create_model("openfoam_recon", recon_srun, path=exp_dir)

    # Start the reconstrucion model
    exp.start(openfoam_recon)

def run_serial_simulation(alloc, foam_env_vars, sim_dir):
    """Run a serial version of the openFOAM simulation

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param sim_dir: The directory where input files are
                    located
    :type sim_dir: str
    """

    # Definte the simulation directory based on the
    # experiment name and the previous model name used
    exp_dir = getcwd() + '/openfoam_ml/openfoam'

    # Create the run settings for the simulation model
    openfoam_srun = SrunSettings(foam_env_vars['FOAM_APPBIN'] + "/simpleFoam_ML",
                                 env_vars = foam_env_vars,
                                 alloc = allocation)
    openfoam_srun.set_nodes(1)
    openfoam_srun.set_tasks(1)

    # Create the simulation model
    openfoam_model = exp.create_model("openfoam_sim", openfoam_srun, path=exp_dir)

    # Set the model to copy input files
    openfoam_model.attach_generator_files(to_copy=[sim_dir])

    # Generate the experiment directory
    exp.generate(openfoam_model, overwrite=True)

    # Start teh simulation model
    exp.start(openfoam_model)

if __name__ == "__main__":

    # Orchestrator settings
    db_node_count = 3
    db_cpus = 36
    db_tpq = 4
    db_port = 6379

    # Simulation settings
    sim_node_count = 1
    sim_dir = "./sim_inputs/pitzDaily_ML/"
    sim_tasks_per_node = 1

    # Training settings
    training_node_count = 1
    training_dir = "./training"
    training_tasks_per_node = 1

    # Model settings
    model_file = "./openfoam_ml/training/ML_SA_CG.pb"
    device = "CPU"
    batch_size = 1

    # Launch orchestrator
    db = start_database(db_port, db_node_count, db_cpus, db_tpq)

    # Retrieve one of the orchestrator addresses to set
    # the ML model into the database
    address = db.get_address()[0]

    # Retrieve OpenFOAM environment variables for execution
    foam_env_vars = get_openfoam_env_vars()

    # Get simulation allocation
    total_nodes = max(sim_node_count, training_node_count)
    allocation = slurm.get_allocation(nodes=total_nodes,
                                      time="10:00:00",
                                      options={"exclusive": None,
                                               "job-name": "openfoam"})

    # Train ML model
    #run_training(allocation, training_dir,
    #             training_node_count,
    #             training_tasks_per_node)

    # Set the trained model into the database
    set_model(model_file, device, batch_size, address, bool(db_node_count>1))

    # Run parallel OpenFOAM if resources are sufficient
    if sim_tasks_per_node > 1 or sim_node_count > 1:
        # Run the decomposition step for a parallel run
        run_decomposition(allocation, foam_env_vars, sim_dir)

        # Run the openFOAM simulation
        run_simulation(allocation, foam_env_vars,
                    sim_node_count, sim_tasks_per_node)

        # Reconstruct the results from a parallel run
        run_reconstruction(allocation, foam_env_vars)
    else:
        # Run a serial version of the simulation
        run_serial_simulation(allocation, foam_env_vars, sim_dir)






