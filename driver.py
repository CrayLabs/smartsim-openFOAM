from smartsim.settings import SrunSettings, AprunSettings
from smartsim.database import SlurmOrchestrator, CobaltOrchestrator
from smartsim import Experiment, slurm, constants
from smartredis import Client
from os import environ, getcwd, listdir, walk, rename
from shutil import copyfile
from os.path import isfile, join
import time
import math

exp_name = "openfoam_ml"
exp = None
launcher = None

def create_of_model(launcher, nodes, ppn,
                    exe, exe_args, model_name,
                    exec_dir, env_vars):
    """Construct an SmartSim Model for the OpenFOAM
    executable.

    :param launcher: The launcher to use for run settings
    :type launcher: str
    :param nodes: The number of nodes to use
                  for the model run settings
    :type nodes: int
    :param ppn: The processes per node for the model
                run settings
    :type ppn: int
    :param exe: The location of the executable (bsolute path)
    :type exe: str
    :param exe_args: The executable arguments
    :type exe_args: dict with key for the argument
                    and value for the argument value
    :param model_name: The name of the model
    :type model_name: str
    :param exec_dir: The directory to associated with
                    model execution.  Can be None.
    :type exec_dir: str
    :param env_vars: A dictionary of environment variables
    :type env_vars: dict
    :return: A SmartSim model
    :rtype: SmartSim Model

    """
    # using slurm/srun
    if launcher == "slurm":
        rs = SrunSettings(exe,
                          exe_args=exe_args,
                          env_vars=env_vars)
        rs.set_nodes(nodes)
        rs.set_tasks_per_node(ppn)
    # using cobalt/aprun
    else:
        rs = AprunSettings(exe, exe_args=exe_args)
        rs.set_tasks(nodes*ppn)
        rs.set_tasks_per_node(ppn)


    if exec_dir is None:
        open_foam = exp.create_model(model_name,
                                     run_settings=rs)
    else:
        open_foam = exp.create_model(model_name,
                                     run_settings=rs,
                                     path=exec_dir)

    return open_foam

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

def start_database(port, nodes, cpus, tpq, interface):
    """Create and start the Redis database

    :param port: port number of database
    :type port: int
    :param nodes: number of database nodes
    :type nodes: int
    :param cpus: number of cpus per node
    :type cpus: int
    :param tpq: number of threads per queue
    :type tpq: int
    :param interface: the network interface to bind to
    :type interface: str
    :return: orchestrator instance
    :rtype: Orchestrator
    """

    if launcher == "slurm":
        db = SlurmOrchestrator(port=port,
                               db_nodes=nodes,
                               batch=False,
                               interface=interface)
    else:
        db = CobaltOrchestrator(port=port,
                                db_nodes=nodes,
                                batch=False,
                                interface=interface)
    db.set_cpus(cpus)
    exp.generate(db)
    exp.start(db)
    return db

def run_decomposition(foam_env_vars, exec_dir,
                      model_prefix="", block=False):
    """Run the OpenFOAM decomposition utility in a
    specified directory.

    :param foam_env_vars: Environment variables
                          needed to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param exec_dir: The directory where decomp should be run
    :type exec_dir: str
    :param model_prefix: A prefix to add to the model
                         name
    :type model_prefix: str
    :param block: Boolean indicating if the decomp
                  should block on execution
    :type block: bool
    """

    # Create a SmartSim model for decomposition utility
    executable = foam_env_vars['FOAM_APPBIN'] + "/decomposePar"
    name = model_prefix + "decomp"
    nodes = 1
    ppn = 1
    exe_args = None
    decomp_model = create_of_model(launcher, nodes, ppn, executable,
                                   exe_args, name, exec_dir, foam_env_vars)

    # Run the openFOAM decomposition utility
    exp.start(decomp_model, block=block)

def run_reconstruction(foam_env_vars, exec_dir,
                       model_prefix="", block=False):
    """Run the openFOAM parallel reconstruction utility

    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param exec_dir: The directory where decomp should be run
    :type exec_dir: str
    :param model_prefix: A prefix to add to the model
                         name
    :type model_prefix: str
    :param block: Boolean indicating if the reconstruction
                  should block on execution
    :type block: bool
    """

    # Create the reconstruction model
    executable = foam_env_vars['FOAM_APPBIN'] + "/reconstructPar"
    name = model_prefix + "recon"
    nodes = 1
    ppn = 1
    exe_args = None
    openfoam_recon = create_of_model(launcher, nodes, ppn, executable,
                                     exe_args, name, exec_dir, foam_env_vars)

    # Start the reconstrucion utility
    exp.start(openfoam_recon, block=block)

def generate_data_gen_files(node_per_case, tasks_per_node,
                            input_dir, name):
    """Generate the OpenFOAM cases used for training data

    :param node_per_case: The number of nodes to
                           use per data generation case
    :type node_per_case: int
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
    n_proc = tasks_per_node * node_per_case

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

def run_data_gen_decomposition(foam_env_vars, dir):
    """Run the decomposition step for the training data
    cases

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
        run_decomposition(foam_env_vars, d,
                          model_prefix=model_prefix, block=False)

    exp.poll()

def run_data_generation(foam_env_vars, node_per_case,
                        tasks_per_node, gen_dir):
    """Run the openFOAM data generation simulations

    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param node_per_case: The number of nodes to
                           use per data generation case
    :type node_per_case: int
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
        # Create the simulation model
        exec_path = "/".join([gen_dir,f"Case{i}"])
        model_name = f"data_gen{i}"
        model = create_of_model(launcher, node_per_case, tasks_per_node,
                                executable, exe_args, model_name, exec_path,
                                foam_env_vars)

        # Start the simulation model
        exp.start(model, block=False)

    exp.poll()

def run_data_gen_reconstruction(foam_env_vars, gen_dir):
    """Run the data generation reconstruction step

    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param gen_dir: The directory where data generation cases
               reside (e.g. Case1, Case2, ...)
    :type gen_dir: str
    """

    case_dirs = ["/".join([gen_dir, f"Case{i}"]) for i in range(1,7)]

    for i, dir in enumerate(case_dirs):
        model_prefix = f"case{i+1}_"
        run_reconstruction(foam_env_vars, dir,
                           model_prefix=model_prefix, block=False)

    exp.poll()

def run_data_gen_dataset_construction(gen_dir):
    """Run the script to aggregate training data

    :param gen_dir: The directory where the generated cases
                    are located (i.e. Case1, Case2, etc..)
    :type gen_dir: str
    """
    # Store the executable and exec args
    executable = "python"
    exe_args = "training_data_maker.py"

    # Create the data aggregation model
    model_name = "dataset_construction"
    nodes = 1
    tasks_per_node = 1
    script_model = create_of_model(launcher, nodes, tasks_per_node,
                                   executable, exe_args, model_name, gen_dir,
                                   foam_env_vars)

    # Start the data aggregation script
    exp.start(script_model)

def run_training(training_dir, training_node_count,
                 training_tasks_per_node, gen_dir):
    """Run the TensorFlow training script

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

    # Create a SmartSim model for the training model
    model_name = "training"
    executable = "python"
    exe_args = "ML_Model.py"
    modle_name = "training"
    nodes = training_node_count
    tasks_per_node = training_tasks_per_node
    training_model = create_of_model(launcher, nodes, tasks_per_node,
                                     executable, exe_args, model_name, None,
                                     foam_env_vars)

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

def run_simulation(foam_env_vars, nodes, ppn, sim_dir):
    """Run the openFOAM simulation

    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param nodes: The number of compute nodes
                           to use for the simulation
    :type nodes: int
    :param ppn: The number of processors per node
    :type ppn: int
    :param sim_dir: The directory where the generated input files
                    are located
    :type sim_dir: str
    """
    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/simpleFoam_ML"

    # Set exec args to "-parallel" if needed
    exe_args = None
    if (nodes*ppn)>1:
        exe_args = "-parallel"

    # Create the simulation model
    model_name = "sim"
    model = create_of_model(launcher, nodes, ppn, executable,
                            exe_args, model_name, sim_dir,
                            foam_env_vars)

    # Start the simulation model
    exp.start(model)

def run_foamtovtk(foam_env_vars, sim_dir):
    """Run the foamToVTK utility to process output
    files into VTK files

    :param foam_env_vars: Environment variables needed
                          to run openFOAM
    :type foam_env_vars: dict of str keys and str values
    :param sim_dir: The directory where the generated simulation
                files are located
    :type sim_dir: str
    """
    # Store the executable as a variable
    executable = foam_env_vars['FOAM_APPBIN'] + "/foamToVTK"

    # Create the reconstruction model
    model_name = "foamtovtk"
    nodes = 1
    ppn = 1
    exe_args = None
    model = create_of_model(launcher, nodes, ppn, executable,
                            exe_args, model_name, sim_dir,
                            foam_env_vars)
    # Start the fomatovtk utility
    exp.start(model)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run OpenFOAM ML Experiment")
    parser.add_argument("--launcher", type=str, default="slurm", help="Launcher for the experiment")
    parser.add_argument("--db_nodes", type=int, default=1, help="Number of nodes for the database")
    parser.add_argument("--db_port", type=int, default=6780, help="Port for the database")
    parser.add_argument("--db_interface", type=str, default="ipogif0", help="Network interface for the database")
    parser.add_argument("--gen_nodes", type=int, default=2, help="Number of nodes to use for each data generation case")
    parser.add_argument("--gen_ppn", type=int, default=24, help="Number of processors per node for each generation case")
    parser.add_argument("--sim_nodes", type=int, default=1, help="Number of nodes for the OpenFOAM inference case")
    parser.add_argument("--sim_ppn", type=int, default=24, help="Number of processors per node for OpenFOAM inference case")
    args = parser.parse_args()

    # Orchestrator settings
    db_node_count = args.db_nodes
    db_port = args.db_port
    db_interface = args.db_interface
    db_cpus = 16
    db_tpq = 4

    # Data generation settings
    gen_nodes_per_case = args.gen_nodes
    gen_tasks_per_node = args.gen_ppn
    gen_input_dir = "./data_generation/"
    gen_name = "data_generation"

    # Simulation settings
    sim_node_count = args.sim_nodes
    sim_tasks_per_node = args.sim_ppn
    sim_input_dir = "./simulation_inputs/"


    # Training settings (do not change)
    training_node_count = 1
    training_tasks_per_node = 1
    training_dir = "./training"

    # Model settings
    model_file = "./" + exp_name + "/training/ML_SA_CG.pb"
    device = "CPU"
    batch_size = 1

    # Script variables
    sim_name = "openfoam"
    sim_dir = "/".join([getcwd(),exp_name,sim_name])
    gen_dir = "/".join([getcwd(),exp_name,gen_name])

    # Create and set the global variable exp
    exp = Experiment(name=exp_name, launcher=args.launcher)

    # Create and set the global variable launcher
    launcher = args.launcher

    # Launch orchestrator
    db = start_database(db_port, db_node_count, db_cpus, db_tpq, db_interface)

    # Retrieve one of the orchestrator addresses to set
    # the ML model into the database
    address = db.get_address()[0]

    # Retrieve OpenFOAM environment variables for execution
    foam_env_vars = get_openfoam_env_vars()

    # Generate the data generation input files
    generate_data_gen_files(gen_nodes_per_case, gen_tasks_per_node,
                            gen_input_dir, gen_name)

    # Run data generation domain decomposition
    if (gen_tasks_per_node * gen_nodes_per_case) > 1:
        run_data_gen_decomposition(foam_env_vars,gen_dir)

    # Run the data generation cases
    run_data_generation(foam_env_vars, gen_nodes_per_case,
                        gen_tasks_per_node,
                        gen_dir)

    # Run the reconstruction step for data generation
    if (gen_tasks_per_node * gen_nodes_per_case) > 1:
        run_data_gen_reconstruction(foam_env_vars, gen_dir)

    # Run the script to create training dataset
    run_data_gen_dataset_construction(gen_dir)

    # Train the ML model for the simulation
    run_training(training_dir,
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
        run_decomposition(foam_env_vars, sim_dir,
                          model_prefix="sim_", block=True)

    # Run the openFOAM simulation
    run_simulation(foam_env_vars,
                   sim_node_count, sim_tasks_per_node, sim_dir)

    # Run reconstruction for parallel execution
    if sim_tasks_per_node * sim_node_count > 1:
        run_reconstruction(foam_env_vars, sim_dir,
                           model_prefix="sim_", block=True)

    # Run foamToVTK to generate VTK output files
    run_foamtovtk(foam_env_vars, sim_dir)





