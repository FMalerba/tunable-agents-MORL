# Introduction
This repository contains the code necessary to train, evaluate and display results for tunable agents in a Multi-Objective Deep Reinforcement Learning scenario. The environment implemented is the Multi-Objective Gathering environment described in [this](https://ala2019.vub.ac.be/papers/ALA2019_paper_12.pdf) paper. The original code repository was [forked](https://github.com/FMalerba/gym-mo) and slightly modified to improve code readability, adapt the classes to interface with this repository's code and change some undesiderable behaviour.

The key difference between the `GatheringWrapper` environment implemented in this repo and the one implemented in the original paper is that here we are interested in agents trained to have non-linear utility functions. 

# Setup
To start, clone the repository with its submodule using the command
```
git clone --recurse-submodules https://github.com/FMalerba/tunable-agents-MORL
```

Install the requirements listed in the `requirements.txt` file and then run the commmand 
```
pip install .
```

From both `tunable-agents-MORL/` and  `tunable-agents-MORL/tunable_agents/environments/gathering_env/gym-mo/` 

This will install the package containing all the environments and required code to run `train_agent.py`, the evaluation scripts and the code in the jupyter notebooks.

# Training an agent

This repository makes extensive use of [Gin Config](https://github.com/google/gin-config) in order to set all the parameters that are needed for the execution of code. Gin Config works by passing configuration files that set all the parameters to be used by functions and classes decorated with `@gin.configurable`. The files are passed to the scripts using the `--gin_files` flag and they are stored under the `configs` folder. To pass multiple files one can simply invoke the `--gin_files` flag multiple times feeding different paths for every configuration file. More information on Gin Config can be found at the project's repository.


In general, `train_agent.py` will expect a `--root_dir` to tell it where to store the trained model and the results for the rewards. Three `--gin_files` flags will also be needed; one to specify the training process, one to sepecify the model to be used and one to specify the environment and utility functions to be used. Additionally the `--gin_bindings` flag can be passed to set a certain parameter (or change it from the value set in the configuration files); this is usually used to assign unique names to different runs of a same experiment. A typical training command will thus look something like this:
```
python tunable-agents-MORL/train_agent.py \
--root_dir='experiments_results' \
--gin_files="tunable-agents-MORL/configs/replication.gin" \
--gin_files="tunable-agents-MORL/configs/qnets/64_64_model.gin" \
--gin_files="tunable-agents-MORL/configs/envs/replication_env.gin" \
--gin_bindings="train_eval.training_id='replication-1'"
```




