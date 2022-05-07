#!/bin/bash
RUNS=5


PREFIX="0_attempt/2022-05-05_00:00:00"
PATH_SRC="src"
PATH_OUTPUT="output/"
PATH_TRAINED_AGENTS="trained_agents"

###############################################################################
mkdir -p $PATH_OUTPUT
mkdir -p $PATH_TRAINED_AGENTS

for (( i=0 ; i<$RUNS ; i++ )); 
do
#    filename_vanilla="${PREFIX}_VPG_vanilla_$i"
#    filename_moderate="${PREFIX}_VPG_moderate_$i"
#    filename_extreme="${PREFIX}_VPG_extreme_$i"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

#    filename_vanilla="${PREFIX}_PPO_vanilla_$i"
#    filename_moderate="${PREFIX}_PPO_moderate_$i"
#    filename_extreme="${PREFIX}_PPO_extreme_$i"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

#    filename_vanilla="${PREFIX}_DDPG_vanilla_$i"
#    filename_moderate="${PREFIX}_DDPG_moderate_$i"
#    filename_extreme="${PREFIX}_DDPG_extreme_$i"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
#    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

    filename_vanilla="${PREFIX}_TD3_vanilla_$i"
    filename_moderate="${PREFIX}_TD3_moderate_$i"
    filename_extreme="${PREFIX}_TD3_extreme_$i"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"

    filename_vanilla="${PREFIX}_SAC_vanilla_$i"
    filename_moderate="${PREFIX}_SAC_moderate_$i"
    filename_extreme="${PREFIX}_SAC_extreme_$i"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_vanilla" --output "${PATH_OUTPUT}/$filename_vanilla"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_moderate" --output "${PATH_OUTPUT}/$filename_moderate"
    python ${PATH_SRC}/test.py --input_condition "vanilla" --agent "${PATH_TRAINED_AGENTS}/$filename_extreme" --output "${PATH_OUTPUT}/$filename_extreme"    
done
