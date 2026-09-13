import os
import sys
import time
import datetime
import pickle
import yaml
import logging

from torch.utils.tensorboard import SummaryWriter
from src.server import Server


# Usage:
# python main_experiment.py path/to/config.yaml
# python main_experiment.py path/to/config.yaml --resume
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "./config_adaptive.yaml"

resume = "--resume" in sys.argv[2:]

print(f"[INFO] Using configuration: {config_file}")
print(f"[INFO] Resume mode: {resume}")

with open(config_file) as c:
    configs = list(yaml.load_all(c, Loader=yaml.FullLoader))


def get_config(key, default=None):
    for config in configs:
        if isinstance(config, dict) and key in config:
            return config[key]
    if default is not None:
        return default
    raise KeyError(f"Configuration section not found: {key}")


global_config = get_config("global_config")
data_config = get_config("data_config")
fed_config = get_config("fed_config")
optim_config = get_config("optim_config")
init_config = get_config("init_config")
model_config = get_config("model_config")
log_config = get_config("log_config")
cl_config = get_config("cl_config")
loss_config = get_config("loss_config")
adaptive_gamma_config = get_config("adaptive_gamma", {})
temp_queue_config = get_config("temp_queue", {})
logit_adjustment_config = get_config("logit_adjustment", {})
global_prototype_config = get_config("global_prototype_config", {})
decorrelation_config = get_config("decorrelation_config", {})


# Give every experiment a unique log directory
experiment_name = os.path.splitext(
    os.path.basename(config_file)
)[0]

timestamp = datetime.datetime.now().strftime(
    "%Y-%m-%d_%H_%M_%S_%f"
)

log_config["log_path"] = os.path.join(
    log_config["log_path"],
    f"{experiment_name}_{timestamp}"
)  

os.makedirs(log_config["log_path"], exist_ok=True)

writer = SummaryWriter(
    log_dir=log_config["log_path"],
    filename_suffix="FL"
)

logging.basicConfig(
    filename=os.path.join(log_config["log_path"], log_config["log_name"]),
    level=logging.INFO,
    format="[%(levelname)s](%(asctime)s) %(message)s",
    datefmt="%Y/%m/%d/ %I:%M:%S %p"
)


message = "\n[WELCOME] Unfolding configurations...!"
print(message)
logging.info(message)

print(f"[INFO] Config file: {config_file}")
logging.info(f"[INFO] Config file: {config_file}")

for config in configs:
    print(config)
    logging.info(config)

print()


central_server = Server(
    writer,
    model_config,
    global_config,
    data_config,
    init_config,
    fed_config,
    optim_config,
    cl_config,
    loss_config,
    adaptive_gamma_config,
    temp_queue_config,
    logit_adjustment_config,
    global_prototype_config
)

# ---------------------------------------------------------
# Checkpoint / resume paths
# ---------------------------------------------------------
checkpoint_dir = os.path.join(
    "checkpoints",
    experiment_name
)

metrics_file = os.path.join(
    "results",
    experiment_name,
    "metrics.csv"
)

central_server.setup(
    decorrelation_config=decorrelation_config,
    global_prototype_config=global_prototype_config
)

central_server.fit(
    resume=resume,
    checkpoint_dir=checkpoint_dir,
    metrics_file=metrics_file
)


with open(
    os.path.join(
        log_config["log_path"],
        f"result_{os.path.basename(os.path.normpath(log_config['log_path']))}.pkl"
    ),
    "wb"
) as f:
    pickle.dump(central_server.results, f)


message = "...done all learning process!\n...exit program!"
print(message)
logging.info(message)

time.sleep(3)
