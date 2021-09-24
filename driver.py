from smartsim.settings import SrunSettings
from smartsim.database import SlurmOrchestrator
from smartsim import Experiment, slurm, constants
from smartredis import Client
from os import environ, getcwd, listdir, walk, rename
from shutil import copyfile
from os.path import isfile, join
import time
import math

exp_name = "openfoam_ml"
exp = Experiment(name=exp_name, launcher="slurm")

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
    """Create and start the Redis database

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
    db.set_walltime("2:00:00")
    db.set_batch_arg("exclusive", None)
    exp.generate(db)
    exp.start(db)
    return db

def run_decomposition(alloc, foam_env_vars, dir,
                      model_prefix="", block=False):
    """Run the OpenFOAM decomposition utility in a
    specified directory.

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables
                          needed to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param dir: The directory where decomp should be run
    :type dir: str
    :param model_prefix: A prefix to add to the model
                         name
    :type model_prefix: str
    :param block: Boolean indicating if the decomp
                  should block on execution
    :type block: bool
    """

    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/decomposePar"

    # Create the run settings for the mesh decomposition
    srun = SrunSettings(exe = executable,
                        env_vars = foam_env_vars,
                        alloc = alloc)
    srun.set_nodes(1)
    srun.set_tasks(1)

    # Create a SmartSim model for decomposition utility
    name = model_prefix + "decomp"
    decomp_model = exp.create_model(name, srun, path=dir)

    # Run the openFOAM decomposition utility
    exp.start(decomp_model, block=block)

def run_reconstruction(alloc, foam_env_vars, dir,
                       model_prefix="", block=False):
    """Run the openFOAM parallel reconstruction utility

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :type dir: str
    :param model_prefix: A prefix to add to the model
                         name
    :type model_prefix: str
    :param block: Boolean indicating if the reconstruction
                  should block on execution
    :type block: bool
    """

    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/reconstructPar"

    # Create the run settings for recombining
    srun = SrunSettings(exe = executable,
                        env_vars = foam_env_vars,
                        alloc = allocation)
    srun.set_nodes(1)
    srun.set_tasks(1)

    # Create the reconstruction model
    name = model_prefix + "recon"
    openfoam_recon = exp.create_model(name, srun, path=dir)

    # Start the reconstrucion utility
    exp.start(openfoam_recon, block=block)

def generate_data_gen_files(node_count, tasks_per_node,
                            input_dir, name):
    """Generate the OpenFOAM cases used for training data

    :param node_count: The number of compute nodes
                       to use for the data generation
    :type node_count: int
    :param tasks_per_node: The number of tasks
                           per compute node
    :type tasks_per_node: int
    :param input_dir: The directory where the cases
                      (i.e. Case1, Case2, Case3..)
                      are located
    :type input_dir: str
    :param name: The name of the data generation model
    :type name: str
    """

    # Calculate the closest near-square values of n_proc
    # In the worst case, 1 x n_proc will be used for
    # decomposition
    n_proc = tasks_per_node

    big = math.ceil(math.sqrt(n_proc))
    small = math.floor(n_proc/big)
    while small * big != float(n_proc):
        big -= 1
        small = math.floor(n_proc/big)

    # Save the processor counts as a single string
    # param for now since multiple tags per line
    # is currently not supported
    params = {
        "proc_x_y":str(big) + " " + str(small),
        "n_procs":str(n_proc)}

    # Create a SmartSim model that will generate all of
    # files for the data generation run.  The generation is
    # split into a copy step and a configure step because
    # currently to_configure param does not support
    # directories and we want to preserve directory structure.
    copy_model = exp.create_model(name, None)

    # Copy all input files
    copy_model.attach_generator_files(to_copy=input_dir)

    # Generate the experiment file directory
    exp.generate(copy_model, overwrite=True)

    # Create a list of tuples containing the
    # original input directory, the subdirectory inside
    # of the top-level diretory and the file name
    config_files = [(input_dir, f"/Case{i}/system/", "decomposeParDict") for i in range(1,7)]

    # Create a model used to write configured input file
    config_model =  exp.create_model("config", None, params=params)

    # Attach all the files to be configured.
    # Because all decomposeParDict file are identical,
    # we only need to configure one and copy it to all
    # directories
    file_name = config_files[0][0] + \
                config_files[0][1] + \
                config_files[0][2]
    config_model.attach_generator_files(to_configure=[file_name])

    # Generate the configured files into a "config" directory
    exp.generate(config_model, tag="@", overwrite=True)

    # Copy the configured files to the simulation directory
    for c_file in config_files:
        old_file = "./" + exp_name + "/config/" + c_file[2]
        new_file = "./" + exp_name + f"/{name}/" + c_file[1] + c_file[2]
        copyfile(old_file, new_file)

def run_data_gen_decomposition(alloc, foam_env_vars, dir):
    """Run the decomposition step for the training data
    cases

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param dir: The directory where the generated
                cases are located (i.e. Case1, Case2, etc..)
    :type dir: str
    """
    case_dirs = ["/".join([dir, f"Case{i}"]) for i in range(1,7)]

    for i, d in enumerate(case_dirs):
        model_prefix = f"case{i+1}_"
        run_decomposition(alloc, foam_env_vars, d,
                          model_prefix=model_prefix, block=False)

    exp.poll()

def run_data_generation(alloc, foam_env_vars, node_count,
                        tasks_per_node, gen_dir):
    """Run the openFOAM data generation simulations

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param node_count: The number of compute nodes available
    :type node_count: int
    :param tasks_per_node: The number of tasks per compute node
    :type tasks_per_node: int
    :param gen_dir: The directory where the generated
                    cases are located (i.e. Case1, Case2, etc..)
    :type gen_dir: str
    """
    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/simpleFoam"

    # Set exec args to "-parallel" if needed
    exe_args = None
    if tasks_per_node>1:
        exe_args = "-parallel"

    for i in range(1,7):
        # Create the run settings for the simulation model
        srun = SrunSettings(exe = executable,
                            exe_args = exe_args,
                            env_vars = foam_env_vars,
                            alloc = allocation)
        srun.set_nodes(1)
        srun.set_tasks(tasks_per_node)

        # Create the simulation model
        exec_path = "/".join([gen_dir,f"Case{i}"])
        model_name = f"data_gen{i}"
        model = exp.create_model(model_name, srun, path=exec_path)

        # Start the simulation model
        exp.start(model, block=False)

    exp.poll()

def run_data_gen_reconstruction(alloc, foam_env_vars, dir):
    """Run the data generation reconstruction step

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param dir: The directory where data generation cases
               reside (e.g. Case1, Case2, ...)
    :type dir: str
    """

    case_dirs = ["/".join([dir, f"Case{i}"]) for i in range(1,7)]

    for i, dir in enumerate(case_dirs):
        model_prefix = f"case{i+1}_"
        run_reconstruction(alloc, foam_env_vars, dir,
                           model_prefix=model_prefix, block=False)

    exp.poll()

def run_data_gen_dataset_construction(alloc, dir):
    """Run the script to aggregate training data

    :param alloc: The allocation on which to run
    :type alloc: str
    :param dir: The directory where the generated cases
                    are located (i.e. Case1, Case2, etc..)
    :type dir: str
    """
    # Store the executable and exec args
    executable = "python"
    exe_args = "training_data_maker.py"

    # Create the run settings
    srun = SrunSettings(exe = executable,
                               exe_args = exe_args,
                               alloc = allocation)
    srun.set_nodes(1)
    srun.set_tasks(1)

    # Create the data aggregation model
    model_name = "dataset_construction"
    script_model = exp.create_model(model_name, srun, path=gen_dir)

    # Start the data aggregation script
    exp.start(script_model)

def run_training(alloc, training_dir, training_node_count,
                 training_tasks_per_node, gen_dir):
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
    :param gen_dir: The directory where data generation cases
                reside
    :type gen_dir: str
    """

    # Create the run settings for the training script
    srun = SrunSettings(exe = "python",
                        exe_args = "ML_Model.py",
                        alloc = alloc)
    srun.set_nodes(training_node_count)
    srun.set_tasks(training_tasks_per_node)

    # Create a SmartSim model for the training model
    training_model = exp.create_model("training", srun)

    # Set the model to copy input files
    files_to_copy = []
    files_to_copy.append(training_dir)
    files_to_copy.append("/".join([gen_dir,"Total_dataset.npy"]))
    files_to_copy.append("/".join([gen_dir,"means"]))

    training_model.attach_generator_files(to_copy=files_to_copy)

    # Generate the experiment directory
    exp.generate(training_model, overwrite=True)

    # Run the training script
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



def generate_simulation_files(node_count, tasks_per_node,
                              sim_input_dir, sim_name, gen_dir):
    """Generate the OpenFOAM simulation directory

    :param node_count: The number of compute nodes
                           to use for the simulation
    :type node_count: int
    :param tasks_per_node: The number of tasks
                               per compute node
    :type tasks_per_node: int
    :param sim_input_dir: The directory where sim input
                          files are located
    :type sim_input_dir: str
    :param sim_name: The name of the simulation (model)
    :type sim_name: str
    :param gen_dir: The directory where data generation cases
                    reside
    :type gen_dir: str
    """

    # Calculate the closest near-square values of n_proc
    # In the worst case, 1 x n_proc will be used for
    # decomposition
    n_proc = node_count * tasks_per_node

    big = math.ceil(math.sqrt(n_proc))
    small = math.floor(n_proc/big)
    while small * big != float(n_proc):
        big -= 1
        small = math.floor(n_proc/big)

    # Save the processor counts as a single string
    # param for now since multiple tags per line
    # is currently not supported
    params = {
        "proc_x_y":str(big) + " " + str(small),
        "n_procs":str(n_proc)}

    # Create a SmartSim model that will generate all of
    # files for the OpenFOAM run.  We do this because OpenFOAM
    # splits up execution of the simulation into different
    # steps (i.e. executables).  We split up the generation
    # into a copy step and a configure step because currently
    # to_configure param does not support directories and we
    # want to preserve directory structure
    copy_model = exp.create_model(sim_name, None)

    # Copy all input files
    files_to_copy = []
    files_to_copy.append(sim_input_dir)
    files_to_copy.append("/".join([gen_dir,"means"]))

    copy_model.attach_generator_files(to_copy=files_to_copy)

    # Generate the experiment file directory
    exp.generate(copy_model, overwrite=True)

    # Create a list of tuples containing the
    # original input directory, the subdirectory inside of the
    # top-level diretory and the file name
    config_files = [(sim_input_dir, "system/", "decomposeParDict")]

    # Create a model used to write configured input files
    config_model =  exp.create_model("config", None, params=params)

    # Attach all the files to be configured
    config_model.attach_generator_files(to_configure=[f[0]+f[1]+f[2] for f in config_files])

    # Generate the configured files into a "config" directory
    exp.generate(config_model, tag="@", overwrite=True)

    # Copy the configured files to the simulation directory
    for c_file in config_files:
        old_file = "./" + exp_name + "/config/" + c_file[2]
        new_file = "./" + exp_name + "/openfoam/" + c_file[1] + c_file[2]
        rename(old_file, new_file)

def run_simulation(alloc, foam_env_vars, node_count,
                   tasks_per_node, sim_dir):
    """Run the openFOAM simulation

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param node_count: The number of compute nodes
                           to use for the simulation
    :type node_count: int
    :param tasks_per_node: The number of tasks
                               per compute node
    :type tasks_per_node: int
    :param sim_dir: The directory where the generated input files
                    are located
    :type sim_dir: str
    """
    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/simpleFoam_ML"

    # Set exec args to "-parallel" if needed
    exe_args = None
    if (node_count*tasks_per_node)>1:
        exe_args = "-parallel"

    # Create the run settings for the simulation model
    srun = SrunSettings(exe = executable,
                        exe_args = exe_args,
                        env_vars = foam_env_vars,
                        alloc = allocation)
    srun.set_nodes(node_count)
    srun.set_tasks(tasks_per_node)

    # Create the simulation model
    model = exp.create_model("sim", srun, path=sim_dir)

    # Start the simulation model
    exp.start(model)

def run_foamtovtk(alloc, foam_env_vars, dir):
    """Run the foamToVTK utility to process output
    files into VTK files

    :param alloc: The allocation on which to run
    :type alloc: str
    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param dir: The directory where the generated simulation
                files are located
    :type dir: str
    """
    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/foamToVTK"

    # Create the run settings for recombining
    srun = SrunSettings(exe = executable,
                            env_vars = foam_env_vars,
                            alloc = allocation)
    srun.set_nodes(1)
    srun.set_tasks(1)

    # Create the reconstruction model
    model = exp.create_model("fomatovtk", srun, path=dir)

    # Start the fomatovtk utility
    exp.start(model)

if __name__ == "__main__":

    # Orchestrator settings
    db_node_count = 1
    db_cpus = 36
    db_tpq = 4
    db_port = 6379

    # Data generation settings
    gen_node_count = 12
    gen_input_dir = "./data_generation/"
    gen_tasks_per_node = 30
    gen_name = "data_generation"

    # Simulation settings
    sim_node_count = 1
    sim_input_dir = "./simulation_inputs/"
    sim_tasks_per_node = 30

    # Training settings
    training_node_count = 1
    training_dir = "./training"
    training_tasks_per_node = 1

    # Model settings
    model_file = "./" + exp_name + "/training/ML_SA_CG.pb"
    device = "CPU"
    batch_size = 1

    # Script variables
    sim_name = "openfoam"
    sim_dir = "/".join([getcwd(),exp_name,sim_name])
    gen_dir = "/".join([getcwd(),exp_name,gen_name])

    # Launch orchestrator
    db = start_database(db_port, db_node_count, db_cpus, db_tpq)

    # Retrieve one of the orchestrator addresses to set
    # the ML model into the database
    address = db.get_address()[0]

    # Retrieve OpenFOAM environment variables for execution
    foam_env_vars = get_openfoam_env_vars()

    # Get simulation allocation
    total_nodes = max(gen_node_count, sim_node_count, training_node_count)
    allocation = slurm.get_allocation(nodes=total_nodes,
                                      time="10:00:00",
                                      options={"exclusive": None,
                                               "job-name": "openfoam"})

    # Generate the data generation input files
    generate_data_gen_files(gen_node_count, gen_tasks_per_node,
                            gen_input_dir, gen_name)

    # Run data generation domain decomposition
    if (gen_tasks_per_node * gen_node_count) > 1:
        run_data_gen_decomposition(allocation, foam_env_vars,
                                   gen_dir)

    # Run the data generation cases
    run_data_generation(allocation, foam_env_vars,
                        gen_node_count, gen_tasks_per_node,
                        gen_dir)

    # Run the reconstruction step for data generation
    if (gen_tasks_per_node * gen_node_count) > 1:
        run_data_gen_reconstruction(allocation, foam_env_vars,
                                    gen_dir)

    # Run the script to create training dataset
    run_data_gen_dataset_construction(allocation, gen_dir)

    # Train the ML model for the simulation
    run_training(allocation, training_dir,
                 training_node_count,
                 training_tasks_per_node, gen_dir)

    # Set the trained model into the database
    set_model(model_file, device, batch_size, address,
              bool(db_node_count>1))

    # Generate simulation files
    generate_simulation_files(sim_node_count, sim_tasks_per_node,
                              sim_input_dir, sim_name, gen_dir)

    # Run decomposition for parallel execution
    if sim_tasks_per_node * sim_node_count > 1:
        run_decomposition(allocation, foam_env_vars, sim_dir,
                          model_prefix="sim_", block=True)

    # Run the openFOAM simulation
    run_simulation(allocation, foam_env_vars,
                   sim_node_count, sim_tasks_per_node, sim_dir)

    # Run reconstruction for parallel execution
    if sim_tasks_per_node * sim_node_count > 1:
        run_reconstruction(allocation, foam_env_vars, sim_dir,
                           model_prefix="sim_", block=True)

    # Run foamToVTK to generate VTK output files
    run_foamtovtk(allocation, foam_env_vars, sim_dir)






