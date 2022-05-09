#!/bin/bash
RUNS=5

PREFIX="2022-05-05_00:00:00"
PATH_SRC="src"
PATH_TRAINED_AGENTS="trained_agents"

###############################################################################
mkdir -p $PATH_TRAINED_AGENTS

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_SAC_vanilla"  --runs $RUNS --SAC --epochs 60
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_SAC_moderate" --runs $RUNS --SAC --epochs 60 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_SAC_extreme"  --runs $RUNS --SAC --epochs 60 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_TD3_vanilla"  --runs $RUNS --TD3 --epochs 60
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_TD3_moderate" --runs $RUNS --TD3 --epochs 60 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_TD3_extreme"  --runs $RUNS --TD3 --epochs 60 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_DDPG_vanilla"  --runs $RUNS --DDPG --epochs 50
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_DDPG_moderate" --runs $RUNS --DDPG --epochs 50 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_DDPG_extreme"  --runs $RUNS --DDPG --epochs 50 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_PPO_vanilla"  --runs $RUNS --PPO --epochs 50
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_PPO_moderate" --runs $RUNS --PPO --epochs 50 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_PPO_extreme"  --runs $RUNS --PPO --epochs 50 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_VPG_vanilla"  --runs $RUNS --VPG --epochs 50
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_VPG_moderate" --runs $RUNS --VPG --epochs 50 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${PREFIX}_VPG_extreme"  --runs $RUNS --VPG --epochs 50 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

