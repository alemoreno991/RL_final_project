import argparse
import numpy as np
import matplotlib.pyplot as plt 
import pickle as pkl
from dcm2euler import dcm2euler

plt.rcParams["figure.figsize"] = (16,10)

parser = argparse.ArgumentParser()
parser.add_argument('--num_runs', type=int, default=2)

parser.add_argument('--time', type=str, default="")
parser.add_argument('--train_condition', type=str, default="moderate")
parser.add_argument('--initial_condition', type=str, default="vanilla")

# CLI to configure the algorithm we wanna train the agent with
parser.add_argument('--SAC',  action='store', default=False, const=True, nargs="?")
parser.add_argument('--TD3',  action='store', default=False, const=True, nargs="?")
parser.add_argument('--DDPG', action='store', default=False, const=True, nargs="?")
parser.add_argument('--PPO',  action='store', default=False, const=True, nargs="?")
parser.add_argument('--VPG',  action='store', default=False, const=True, nargs="?")
args = parser.parse_args()

ALGORITHMS = []
if args.SAC:
    ALGORITHMS.append("SAC")
if args.TD3:
    ALGORITHMS.append("TD3")
if args.DDPG:
    ALGORITHMS.append("DDPG")
if args.PPO:
    ALGORITHMS.append("PPO")
if args.VPG:
    ALGORITHMS.append("VPG")

NUMBER_RUNS = args.num_runs
NUMBER_TESTS = 4

#------------------------------------------------------------------------------
def process_test(test):
    actions = test["actions"]
    states  = test["states"]

    a = np.array([])
    for action in actions:
        if len(a)==0:
            a = np.hstack( (a, np.array(action)) ) 
        else:
            a = np.vstack( (a, np.array(action)) )

    s = np.array([])
    for state in states:
        if len(s)==0:
            s = np.hstack( (s, np.array(state)) ) 
        else:
            s = np.vstack( (s, np.array(state)) )

    return a, s

#------------------------------------------------------------------------------
def process_test_over_runs(actions_over_runs, states_over_runs):
    min_steps = np.infty
    for actions in actions_over_runs:
        if actions.shape[0] < min_steps:
            min_steps = actions.shape[0]

    a = np.array([]) 
    for actions in actions_over_runs:
        if len(a)==0:
            a = actions[:min_steps-1,:] 
        else:
            a = np.dstack((a, actions[:min_steps-1,:]))

    s = np.array([]) 
    for states in states_over_runs:
        if len(s)==0:
            s = states[:min_steps-1,:]
        else:
            s = np.dstack((s, states[:min_steps-1,:]))        
    
    return a, s

#------------------------------------------------------------------------------    
solutions_tests = {}
for algorithm in ALGORITHMS:
    tests = {}
    for test in range(NUMBER_TESTS):
        tests["test{}".format(test)] = {} 
    # The information comming from the tests could be of variable lenght
    # Therefore, it's much more involved to parse it
    for test in range(NUMBER_TESTS):
        actions = []
        states = []
        for run in range(NUMBER_RUNS):
            filepath = "./output/{}_{}_{}_{}_{}.pkl".format(args.time, algorithm, args.train_condition, run, args.initial_condition )
            with open(filepath, 'rb') as f:
                obj = pkl.load(f)
            
            a, s = process_test( obj["test{}".format(test)] )
            actions.append(a)
            states.append(s)

        tests["test{}".format(test)]["actions"], tests["test{}".format(test)]["states"] = process_test_over_runs(actions, states)

    solutions_tests[algorithm] = tests

###############################################################################
#   Create the statistical plot for state (with different initial conditions)
###############################################################################
color = [
    "blue",
    "red",
    "green"
]


def runPlotter(data, height, width, subplot_idx, legend, y_label,filename,append=False,color='blue',ax=None,fig=None):
    if not append:
        fig, ax = plt.subplots(height,width)

    for i in range(height*width):
        mu = data.mean(axis=2)
        sigma = data.std(axis=2)
        t = np.arange(len(mu[:,i]))

        ax[ subplot_idx[i] ].plot(t, mu[:,i], lw=2, label=legend, color=color)
        ax[ subplot_idx[i] ].fill_between(t, mu[:,i]+sigma[:,i], mu[:,i]-sigma[:,i], facecolor=color, alpha=0.25)
        ax[ subplot_idx[i] ].legend(loc='upper left')
        ax[ subplot_idx[i] ].set_xlabel('num steps')
        ax[ subplot_idx[i] ].set_ylabel(y_label[i])
        ax[ subplot_idx[i] ].grid()
    
    fig.savefig(filename)

    return fig, ax
#------------------------------------------------------------------------------
   
subplot_idx = [
    (0,0), (0,1), (0,2), 
    (1,0), (1,1), (1,2), 
    (2,0), (2,1), (2,2),
    (3,0), (3,1), (3,2)
]

legend = ALGORITHMS

y_label = [
    "Error Pos_X [m]",
    "Error Pos_Y [m]",
    "Error Pos_Z [m]",
    "Vel_X [m/s]",
    "Vel_Y [m/s]",
    "Vel_Z [m/s]",
    "Roll [deg]",
    "Pitch [deg]",
    "Yaw [deg]",
    "Angular rate - X [deg/s]",
    "Angular rate - Y [deg/s]",
    "Angular rate - Z [deg/s]"
]

axHandle  = [None]*NUMBER_TESTS
figHandle  = [None]*NUMBER_TESTS
flag_append = [False]*NUMBER_TESTS
for algo_idx, algorithm in enumerate(ALGORITHMS):
    tests = solutions_tests[algorithm]
    for test in range(NUMBER_TESTS):
        data = tests["test{}".format(test)]["states"]
        
        pos = data[:,:3,:]
        RBI = data[:,3:12,:]
        vel = data[:,12:15,:]
        omegaB = np.rad2deg(data[:,15:,:])
        
        data = np.hstack((pos,vel,vel,omegaB))
        for i in range(RBI.shape[0]):
            for k in range(NUMBER_RUNS):
                data[i,6:9,k] = np.rad2deg(dcm2euler(RBI[i,:,k].reshape(3,3)))

        figHandle[test], axHandle[test] = runPlotter(
            data, 
            height=4,
            width=3, 
            subplot_idx=subplot_idx,
            legend=algorithm,
            y_label=y_label,
            color=color[algo_idx],
            filename="img/{}_{}_states_{}_{}.png".format(args.time, args.train_condition, test, args.initial_condition),
            append=flag_append[test],
            ax=axHandle[test],
            fig=figHandle[test]
        )
        flag_append[test] = True


###############################################################################
#   Create the statistical plot for state (with different initial conditions)
###############################################################################

subplot_idx = [
    (0,0), (0,1), 
    (1,0), (1,1), 
]

legend = ALGORITHMS

y_label = [
    "Action on Motor_1",
    "Action on Motor_2",
    "Action on Motor_3",
    "Action on Motor_4",
]

axHandle  = [None]*NUMBER_TESTS
figHandle = [None]*NUMBER_TESTS
for algo_idx, algorithm in enumerate(ALGORITHMS):
    data = solutions_tests[algorithm]
    append_flag = not (algo_idx == 0)

    for test in range(NUMBER_TESTS):
        data = tests["test{}".format(test)]["actions"]
        figHandle[test], axHandle[test] = runPlotter(
            data, 
            height=2,
            width=2, 
            subplot_idx=subplot_idx,
            legend=legend[algo_idx],
            y_label=y_label,
            color=color[algo_idx],
            filename="img/{}_{}_actions_{}_{}.png".format(args.time, args.train_condition, test, args.initial_condition),
            append=append_flag,
            ax=axHandle[test],
            fig=figHandle[test]
        )


###############################################################################
#   Load returns
###############################################################################

# solutions_returns = {}
# for algorithm in ALGORITHMS:
#     returns = np.array([])
#     # The information comming from the training is of fixed lenght
#     for run in range(NUMBER_RUNS):
#         filepath = "./data/{}_run{}_{}.pkl".format(algorithm,run,file_suffix)
#         with open(filepath, 'rb') as f:
#             obj = pkl.load(f)

#         if len(returns)==0:
#             returns = np.hstack((returns, np.array(obj["returns"])))
#         else:
#             returns = np.vstack((returns, np.array(obj["returns"])))

#     solutions_returns[algorithm] = returns

###############################################################################
#   Create the statistical plot for cummulative reward
###############################################################################
# for algo_idx, algorithm in enumerate(ALGORITHMS):
#     returns = solutions_returns[algorithm]

#     mu = returns.mean(axis=0)
#     sigma = returns.std(axis=0)
#     t = np.arange(len(mu))

#     if algo_idx == 0:
#         fig, ax = plt.subplots(1)
#     ax.plot(t, mu, lw=2, label=algorithm, color=color[algo_idx])
#     ax.fill_between(t, mu+sigma, mu-sigma, facecolor=color[algo_idx], alpha=0.25)
#     ax.set_title("Statistical analysis of cummulative reward per episode")
#     ax.legend(loc='upper left')
#     ax.set_xlabel('episodes')
#     ax.set_ylabel('Cummulative reward')
#     ax.grid()

#     plt.savefig("./img/returns_{}.png".format(file_suffix))