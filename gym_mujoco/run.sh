#!/bin/bash
RUNS=10

TIME=`date +%Y-%m-%d_%T`
TIME=2022-05-06_03:35:43
PATH_SRC="src"
PATH_INPUT="input"
PATH_OUTPUT="output"
PATH_TRAINED_AGENTS="trained_agents"

mkdir -p $PATH_OUTPUT
mkdir -p $PATH_TRAINED_AGENTS

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_SAC_vanilla"  --runs $RUNS --SAC --epochs 40
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_SAC_moderate" --runs $RUNS --SAC --epochs 50 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_SAC_extreme"  --runs $RUNS --SAC --epochs 60 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_TD3_vanilla"  --runs $RUNS --TD3 --epochs 1
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_TD3_moderate" --runs $RUNS --TD3 --epochs 1 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_TD3_extreme"  --runs $RUNS --TD3 --epochs 1 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_DDPG_vanilla"  --runs $RUNS --DDPG --epochs 160
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_DDPG_moderate" --runs $RUNS --DDPG --epochs 200 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_DDPG_extreme"  --runs $RUNS --DDPG --epochs 250 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_PPO_vanilla"  --runs $RUNS --PPO --epochs 160
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_PPO_moderate" --runs $RUNS --PPO --epochs 200 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_PPO_extreme"  --runs $RUNS --PPO --epochs 250 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5

python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_VPG_vanilla"  --runs $RUNS --VPG --epochs 300
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_VPG_moderate" --runs $RUNS --VPG --epochs 400 --randomize_reset --disorient
python ${PATH_SRC}/train.py --filename "${PATH_TRAINED_AGENTS}/${TIME}_VPG_extreme"  --runs $RUNS --VPG --epochs 500 --randomize_reset --disorient --init_max_vel 2.0 --init_max_angular_vel 1.0 --init_max_attitude 1.5


###############################################################################
for (( i=0 ; i<$RUNS ; i++ )); 
do
    filename_vanilla="${TIME}_VPG_vanilla_$i"
    filename_moderate="${TIME}_VPG_moderate_$i"
    filename_extreme="${TIME}_VPG_extreme_$i"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

    filename_vanilla="${TIME}_PPO_vanilla_$i"
    filename_moderate="${TIME}_PPO_moderate_$i"
    filename_extreme="${TIME}_PPO_extreme_$i"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

    filename_vanilla="${TIME}_DDPG_vanilla_$i"
    filename_moderate="${TIME}_DDPG_moderate_$i"
    filename_extreme="${TIME}_DDPG_extreme_$i"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

    filename_vanilla="${TIME}_TD3_vanilla_$i"
    filename_moderate="${TIME}_TD3_moderate_$i"
    filename_extreme="${TIME}_TD3_extreme_$i"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla.pkl"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate.pkl"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme.pkl"

    filename_vanilla="${TIME}_SAC_vanilla_$i"
    filename_moderate="${TIME}_SAC_moderate_$i"
    filename_extreme="${TIME}_SAC_extreme_$i"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input "${PATH_INPUT}/vanilla_initial_conditions.json" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"    
done

python $PATH_SRC/plot.py --num_runs $RUNS --time $TIME --condition vanilla --TD3 --SAC --PPO --DDPG --VPG 
python $PATH_SRC/plot.py --num_runs $RUNS --time $TIME --condition moderate --TD3 --SAC --PPO --DDPG --VPG 
python $PATH_SRC/plot.py --num_runs $RUNS --time $TIME --condition extreme --TD3 --SAC --PPO --DDPG --VPG  