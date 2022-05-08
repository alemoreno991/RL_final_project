#!/bin/bash
RUNS=5

PREFIX="0_attempt/2022-05-05_00:00:00"
PATH_SRC="src"
PATH_OUTPUT="output"
PATH_TRAINED_AGENTS="trained_agents"
PATH_IMG="img"


###############################################################################
mkdir -p "$PATH_IMG/0_attempt/"
mkdir -p $PATH_OUTPUT
mkdir -p $PATH_TRAINED_AGENTS

###############################################################################
#------------------------------------------------------------------------------
# Plot returns
#------------------------------------------------------------------------------
python ~/spinningup/spinup/utils/plot.py trained_agents/0_attempt/2022-05-05_00\:00\:00_VPG --xaxis Epoch --value AverageEpRet
python ~/spinningup/spinup/utils/plot.py trained_agents/0_attempt/2022-05-05_00\:00\:00_DDPG --xaxis Epoch --value AverageEpRet
python ~/spinningup/spinup/utils/plot.py trained_agents/0_attempt/2022-05-05_00\:00\:00_PPO --xaxis Epoch --value AverageEpRet
python ~/spinningup/spinup/utils/plot.py trained_agents/0_attempt/2022-05-05_00\:00\:00_TD3 --xaxis Epoch --value AverageEpRet
python ~/spinningup/spinup/utils/plot.py trained_agents/0_attempt/2022-05-05_00\:00\:00_SAC --xaxis Epoch --value AverageEpRet

###############################################################################
#------------------------------------------------------------------------------
# PLOT different training conditions on a vanilla initial condition
#------------------------------------------------------------------------------
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition vanilla  --initial_condition vanilla --TD3 --SAC 
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition moderate --initial_condition vanilla --TD3 --SAC 
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition extreme --initial_condition vanilla  --TD3 --SAC

#------------------------------------------------------------------------------
# PLOT {moderate and extreme} training conditions on a moderate initial condition
#------------------------------------------------------------------------------
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition moderate --initial_condition moderate --TD3 --SAC
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition extreme --initial_condition moderate  --TD3 --SAC

#------------------------------------------------------------------------------
# PLOT {moderate and extreme} training conditions on a extreme initial condition
#------------------------------------------------------------------------------
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition moderate --initial_condition extreme --TD3 --SAC 
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition extreme --initial_condition extreme --SAC --TD3