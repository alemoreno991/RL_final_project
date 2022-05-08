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

python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition vanilla  --initial_condition vanilla --TD3 --SAC #--PPO --DDPG --VPG 
python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition moderate --initial_condition vanilla --TD3 --SAC #--PPO --DDPG --VPG 
#python $PATH_SRC/plot.py --num_runs $RUNS --time $PREFIX --train_condition extreme --initial_condition vanilla  --TD3 --SAC #--PPO --DDPG --VPG  
