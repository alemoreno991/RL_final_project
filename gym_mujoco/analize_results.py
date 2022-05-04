import numpy as np
import matplotlib.pyplot as plt 
import pickle as pkl
plt.rcParams["figure.figsize"] = (16,10)

NUMBER_RUNS = 10
NUMBER_TESTS = 4

file_suffix = "_2022-05-04_04:35:36.pkl"
###############################################################################
#   Load information
###############################################################################
returns = np.array([])
# The information comming from the training is of fixed lenght
for run in range(NUMBER_RUNS):
    filepath = "./data/SAC_run{}{}".format(run,file_suffix)
    with open(filepath, 'rb') as f:
        obj = pkl.load(f)

    if not np.any(returns):
        returns = np.hstack((returns, np.array(obj["returns"])))
    else:
        returns = np.vstack((returns, np.array(obj["returns"])))

#------------------------------------------------------------------------------
def process_test(test):
    actions = test["actions"]
    states  = test["states"]

    a = np.array([])
    for action in actions:
        if not np.any(a):
            a = np.hstack( (a, np.array(action)) ) 
        else:
            a = np.vstack( (a, np.array(action)) )

    s = np.array([])
    for state in states:
        if not np.any(s):
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
        if not np.any(a):
            a = actions[:min_steps-1,:] 
        else:
            a = np.dstack((a, actions[:min_steps-1,:]))

    s = np.array([]) 
    for states in states_over_runs:
        if not np.any(s):
            s = states[:min_steps-1,:]
        else:
            s = np.dstack((s, states[:min_steps-1,:]))        
    
    return a, s

#------------------------------------------------------------------------------    

tests = {}
for test in range(NUMBER_TESTS):
    tests["test{}".format(test)] = {} 
# The information comming from the tests could be of variable lenght
# Therefore, it's much more involved to parse it
for test in range(NUMBER_TESTS):
    actions = []
    states = []
    for run in range(NUMBER_RUNS):
        filepath = "./data/SAC_run{}{}".format(run,file_suffix)
        with open(filepath, 'rb') as f:
            obj = pkl.load(f)
        
        a, s = process_test( obj["test{}".format(test)] )
        actions.append(a)
        states.append(s)

    tests["test{}".format(test)]["actions"], tests["test{}".format(test)]["states"] = process_test_over_runs(actions, states)

###############################################################################
#   Create the statistical plot for cummulative reward
###############################################################################
mu = returns.mean(axis=0)
sigma = returns.std(axis=0)
t = np.arange(len(mu))

fig, ax = plt.subplots(1)
ax.plot(t, mu, lw=2, label='mean cummulative reward', color='blue')
ax.fill_between(t, mu+sigma, mu-sigma, facecolor='blue', alpha=0.5)
ax.set_title("Statistical analysis of cummulative reward per episode")
ax.legend(loc='upper left')
ax.set_xlabel('episodes')
ax.set_ylabel('Cummulative reward')
ax.grid()

plt.show()


###############################################################################
#   Create the statistical plot for state (with different initial conditions)
###############################################################################
def runPlotter(data, height, width, subplot_idx, legend, y_label):
    fig, ax = plt.subplots(height,width)
    for i in range(height*width):
        mu = data.mean(axis=2)
        sigma = data.std(axis=2)
        t = np.arange(len(mu[:,i]))

        ax[ subplot_idx[i] ].plot(t, mu[:,i], lw=2, label=legend[i], color='blue')
        ax[ subplot_idx[i] ].fill_between(t, mu[:,i]+sigma[:,i], mu[:,i]-sigma[:,i], facecolor='blue', alpha=0.5)
        ax[ subplot_idx[i] ].legend(loc='upper left')
        ax[ subplot_idx[i] ].set_xlabel('num steps')
        ax[ subplot_idx[i] ].set_ylabel(y_label[i])
        ax[ subplot_idx[i] ].grid()
    
    plt.show()

    return fig, ax
#------------------------------------------------------------------------------
   
subplot_idx = [
    (0,0), (0,1), (0,2), 
    (1,0), (1,1), (1,2), 
    (2,0), (2,1), (2,2),
    (3,0), (3,1), (3,2)
]

legend = [
    "Mean P_x",
    "Mean P_y",
    "Mean P_z",
    "Mean V_x",
    "Mean V_y",
    "Mean V_z",
    "Mean roll",
    "Mean pitch",
    "Mean yaw",
    "Mean \Omega_x",
    "Mean \Omega_y",
    "Mean \Omega_z",
]

y_label = [
    "Position - X",
    "Position - Y",
    "Position - Z",
    "Velocity - X",
    "Velocity - Y",
    "Velocity - Z",
    "Roll",
    "Pitch",
    "Yaw",
    "Angular rate - X",
    "Angular rate - Y",
    "Angular rate - Z"
]
for test in range(NUMBER_TESTS):
    data = tests["test{}".format(test)]["states"]
    fig, ax = runPlotter(
        data, 
        height=4,
        width=3, 
        subplot_idx=subplot_idx,
        legend=legend,
        y_label=y_label
    )

###############################################################################
#   Create the statistical plot for state (with different initial conditions)
###############################################################################

subplot_idx = [
    (0,0), (0,1), 
    (1,0), (1,1), 
]

legend = [
    "Mean action_1",
    "Mean action_2",
    "Mean action_3",
    "Mean action_4",
]

y_label = [
    "Action on Motor_1",
    "Action on Motor_2",
    "Action on Motor_3",
    "Action on Motor_4",
]
for test in range(NUMBER_TESTS):
    data = tests["test{}".format(test)]["actions"]
    fig, ax = runPlotter(
        data, 
        height=2,
        width=2, 
        subplot_idx=subplot_idx,
        legend=legend,
        y_label=y_label
    )