import os
import matplotlib.pyplot as plt
import pickle
import subprocess
import rospkg
import dcargs



def save_and_or_show_plot(directory, name, show_plots):
    if directory is not None:
        plt.savefig(os.path.join(directory, name), dpi=500)
    if show_plots:
        plt.show()


def get_git_revision_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


def get_git_revision_short_hash(directory=None):
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=directory).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return "Failed to get version in get_git_revision_hash()"


def dump_config(config, file_path):
    yaml_str = dcargs.to_yaml(config)
    with open(file_path+".yaml", "w") as f:
        f.write(yaml_str)


def load_config(config_class, file_path):
    with open(file_path + ".yaml", "r") as f:
        config = dcargs.from_yaml(config_class, f)
    return config


def dump_python_object(o, file_path, protocol=5):
    with open(file_path + ".pickle"+str(protocol), "wb") as f:
        pickle.dump(o, f, protocol=protocol)


def load_python_object(file_path, protocol=5):
    with open(file_path + ".pickle"+str(protocol), "rb") as f:
        o = pickle.load(f)
    return o


def get_conda_environment_string(env_name):
    return subprocess.check_output("conda env export --name "+env_name+" | grep -v \"^prefix: \"", shell=True)

