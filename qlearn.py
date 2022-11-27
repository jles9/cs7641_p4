
import pdb


from hiivemdptoolbox.hiive.mdptoolbox.example import openai

from hiive.mdptoolbox.mdp import QLearning
from hiive.mdptoolbox.example import forest
import matplotlib.pyplot as plt

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

import time
import random
import numpy as np

from collections import deque, namedtuple


def plot_basic_exp(data, problemName):

    rewards = []
    errors = []
    times = []
    mean_vs = []
    iterations = []
    max_vs = []


    for datapoint in data:
        # pdb.set_trace()
        # print(datapoint["Reward"])
        rewards.append(datapoint["Reward"])
        errors.append(datapoint["Error"])
        times.append(datapoint["Time"])
        mean_vs.append(datapoint["Mean V"])
        max_vs.append(datapoint["Max V"])
        iterations.append(datapoint["Iteration"])
        #pdb.set_trace()

    plt.style.use('bmh')
    
    #pdb.set_trace()

    avg_rewards = []
    avg_itrs = []
    reward_window = deque(maxlen=500)
    itr_window = deque(maxlen=500)
    for reward in rewards:
        reward_window.append(reward)
        avg_rewards.append(np.mean(reward_window))

    for itr in iterations:
        itr_window.append(itr)
        avg_itrs.append(np.median(itr_window))


    avg_index = np.arange(0,len(avg_rewards))

    plt.figure()
    plt.plot(avg_itrs, avg_rewards, label='average reward', color='blue')
    plt.xlabel("Median Iteration")
    plt.ylabel("Reward")
    plt.legend()
    #plt.show()
    plt.title("Rolling Average Reward for {}".format(problemName))
    plt.savefig('./graphs/QLearn/{}_rewards.png'.format(problemName))

    plt.figure()
    plt.plot(iterations, mean_vs, label='mean v', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend()
    #plt.show()
    plt.title("Mean V per Iteration for {}".format(problemName))
    plt.savefig('./graphs/QLearn/{}_meanV.png'.format(problemName))

    plt.figure()
    plt.plot(iterations, max_vs, label='max v', color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}".format(problemName))
    plt.savefig('./graphs/QLearn/{}_maxV.png'.format(problemName))

    plt.figure()
    plt.plot(iterations, errors, label='errors', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    #plt.show()
    plt.title("Error per Iteration for {}".format(problemName))
    plt.savefig('./graphs/QLearn/{}_error.png'.format(problemName))
    


def plot_sizes(data, problemName):
    
    sizes = []
    total_rewards = []
    sizes_as_string = []
    itrs_total = []

    for size in data.keys():
        result = data[size]
        rewards = []
        iterations = []
        for datapoint in result:
            rewards.append(datapoint["Reward"])
            iterations.append(datapoint["Iteration"])

        total_rewards.append(np.sum(rewards))
        sizes.append(size)
        itrs_total.append(iterations[-1])

    for size in sizes:
        sizes_as_string.append("{}".format(size))
    
    plt.style.use('bmh')

    plt.figure()
    plt.bar(sizes_as_string, total_rewards)
    plt.xlabel("State Size")
    plt.ylabel("Cummulative Reward")
    # plt.legend()
    #plt.show()
    plt.title("Cummulative Rewards for {}, Varying State Size".format(problemName))
    plt.savefig('./graphs/QLearn/{}_size.png'.format(problemName))


def plot_gamma_exp(data, problemName):
    
    gammas = []
    max_vs_total = []
    itrs_total = []

    for gamma in data.keys():
        result = data[gamma]
        max_vs = []
        iterations = []
        for datapoint in result:
            max_vs.append(datapoint["Max V"])
            iterations.append(datapoint["Iteration"])

        gammas.append(gamma)
        itrs_total.append(iterations)
        max_vs_total.append(max_vs)

    
    plt.style.use('bmh')

    plt.figure()

    for index in range(len(gammas)):
        plt.plot(itrs_total[index], max_vs_total[index], label=gammas[index])
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}, Varying gamma".format(problemName))
    plt.savefig('./graphs/QLearn/{}_maxV_gamma.png'.format(problemName))

def plot_alpha_exp(data, problemName):
    
    alphas = []
    max_vs_total = []
    itrs_total = []

    for alpha in data.keys():
        result = data[alpha]
        max_vs = []
        iterations = []
        for datapoint in result:
            max_vs.append(datapoint["Max V"])
            iterations.append(datapoint["Iteration"])

        alphas.append(alpha)
        itrs_total.append(iterations)
        max_vs_total.append(max_vs)

    
    plt.style.use('bmh')

    plt.figure()

    for index in range(len(alphas)):
        plt.plot(itrs_total[index], max_vs_total[index], label=alphas[index])
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}, Varying alpha".format(problemName))
    plt.savefig('./graphs/QLearn/{}_maxV_alpha.png'.format(problemName))


def plot_alpha_decay_exp(data, problemName):
    
    alpha_decs = []
    max_vs_total = []
    itrs_total = []

    for alpha_dec in data.keys():
        result = data[alpha_dec]
        max_vs = []
        iterations = []
        for datapoint in result:
            max_vs.append(datapoint["Max V"])
            iterations.append(datapoint["Iteration"])

        alpha_decs.append(alpha_dec)
        itrs_total.append(iterations)
        max_vs_total.append(max_vs)

    
    plt.style.use('bmh')

    plt.figure()

    for index in range(len(alpha_decs)):
        plt.plot(itrs_total[index], max_vs_total[index], label=alpha_decs[index])
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}, Varying alpha_dec".format(problemName))
    plt.savefig('./graphs/QLearn/{}_maxV_alpha_dec.png'.format(problemName))

def plot_eps_decay_exp(data, problemName):
    
    eps_decs = []
    max_vs_total = []
    itrs_total = []

    for eps_dec in data.keys():
        result = data[eps_dec]
        max_vs = []
        iterations = []
        for datapoint in result:
            max_vs.append(datapoint["Max V"])
            iterations.append(datapoint["Iteration"])

        eps_decs.append(eps_dec)
        itrs_total.append(iterations)
        max_vs_total.append(max_vs)

    
    plt.style.use('bmh')

    plt.figure()

    for index in range(len(eps_decs)):
        plt.plot(itrs_total[index], max_vs_total[index], label=eps_decs[index])
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}, Varying eps_dec".format(problemName))
    plt.savefig('./graphs/QLearn/{}_maxV_eps_dec.png'.format(problemName))


def runForestExp():
    # This will be the "large" problem (>200 states)

    # define some default hyperparams
    gamma = .998


    # Starting with defaults
    # https://pymdptoolbox.readthedocs.io/en/latest/api/example.html
    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    QLearn_agent = QLearning(
        transitions=P,
        reward=R,
        gamma=gamma,
        alpha=.7,
        alpha_decay=.9,
        epsilon=1,
        epsilon_decay=.87,
        n_iter=1000000
    )

    start = time.time()
    result = QLearn_agent.run()
    runtime = time.time() - start

    cuts = np.sum(QLearn_agent.policy)
    wait = len(QLearn_agent.policy) - cuts

    print("Forest Basic Runtime Qlearn: {} \n".format(runtime))
    print(("Forest Basic Policy Cuts QLearn: {} \n".format(cuts)))
    print(("Forest Basic Policy Waits QLearn: {} \n".format(wait)))
    plot_basic_exp(result, problemName="forest")


def runFrozenLakeExp():
    # This will be the "large" problem (>200 states)

    # define default hyperparams
    gamma = .3


    # Starting with defaults
    # https://pymdptoolbox.readthedocs.io/en/latest/api/example.html

    random_map = generate_random_map(size=8, p=0.8)

    P, R = openai(
        env_name="FrozenLake-v0",
        render=False,
        desc=random_map
    )

    QLearn_agent = QLearning(
        transitions=P,
        reward=R,
        gamma=gamma,
        alpha=.8,
        alpha_decay=.99,
        epsilon=1,
        epsilon_decay=.92,
        n_iter=1000000
    )

    start = time.time()
    result = QLearn_agent.run()
    runtime = time.time() - start

    plot_basic_exp(result, problemName="frozenLake")
    print("Frozen Lake Basic Runtime Qlearn: {} \n".format(runtime))


def runSizeForest():

    gamma = 0.998

    sizes = [5, 10, 25, 50, 100, 200, 500, 1000]

    results = {}


    for size in sizes:
        P, R = forest(
            size,
            4,
            2,
            0.1
        )

        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=gamma,
            alpha=.8,
            alpha_decay=.99,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[size] = result

    plot_sizes(results,problemName="Forest")


###################### GAMMA #####################################
def runGammaForest():

    gammas = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    for gamma in gammas:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=gamma,
            alpha=.8,
            alpha_decay=.99,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[gamma] = result

    plot_gamma_exp(results,problemName="Forest")

def runGammaFrozen():

    gammas = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    random_map = generate_random_map(size=8, p=0.8)

    P, R = openai(
        env_name="FrozenLake-v0",
        render=False,
        desc=random_map
    )

    for gamma in gammas:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=gamma,
            alpha=.8,
            alpha_decay=.9,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[gamma] = result

    plot_gamma_exp(results,problemName="FrozenLake")
###################### END GAMMA #####################################


###################### Alpha #####################################
def runAlphaForest():

    alphas = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    for alpha in alphas:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.3,
            alpha=alpha,
            alpha_decay=.95,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=10000
        )

        result = QLearn_agent.run()
        results[alpha] = result

    plot_alpha_exp(results,problemName="Forest")

def runAlphaFrozen():

    alphas = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    random_map = generate_random_map(size=8, p=0.8)

    P, R = openai(
        env_name="FrozenLake-v0",
        render=False,
        desc=random_map
    )

    for alpha in alphas:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.998,
            alpha=alpha,
            alpha_decay=.98,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[alpha] = result

    plot_alpha_exp(results,problemName="FrozenLake")
###################### END Alpha #####################################

###################### Alpha Decay #####################################
def runAlphaDecayForest():

    alpha_decs = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    for alpha_dec in alpha_decs:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.3,
            alpha=.8,
            alpha_decay=alpha_dec,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=10000
        )

        result = QLearn_agent.run()
        results[alpha_dec] = result

    plot_alpha_decay_exp(results,problemName="Forest")


def runAlphaDecayFrozen():

    alpha_decs = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    random_map = generate_random_map(size=8, p=0.8)

    P, R = openai(
        env_name="FrozenLake-v0",
        render=False,
        desc=random_map
    )

    for alpha_dec in alpha_decs:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.998,
            alpha=.98,
            alpha_decay=alpha_dec,
            epsilon=1,
            epsilon_decay=.92,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[alpha_dec] = result

    plot_alpha_decay_exp(results,problemName="FrozenLake")
###################### END Alpha Decay #####################################

###################### Eps Decay #####################################
def runEpsDecayForest():

    eps_decs = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    for eps_dec in eps_decs:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.3,
            alpha=.8,
            alpha_decay=.99,
            epsilon=1,
            epsilon_decay=eps_dec,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[eps_dec] = result

    plot_eps_decay_exp(results,problemName="Forest")

def runEpsDecayFrozen():

    eps_decs = [0.3,0.5,0.7,0.8,0.95,0.97,0.98,0.99]

    results = {}

    P, R = forest(
        1000,
        4,
        2,
        0.1
    )

    for eps_dec in eps_decs:
        QLearn_agent = QLearning(
            transitions=P,
            reward=R,
            gamma=0.3,
            alpha=.98,
            alpha_decay=.99,
            epsilon=1,
            epsilon_decay=eps_dec,
            n_iter=1000000
        )

        result = QLearn_agent.run()
        results[eps_dec] = result

    plot_eps_decay_exp(results,problemName="FrozenLake")
###################### END Eps Decay #####################################



def run_QLearn_exp():
    # runForestExp()
    # runFrozenLakeExp()

    # print("Running Gamma Exp Qlearn")
    # runGammaForest()
    # runGammaFrozen()

    # print("Running Alpha Exp Qlearn")
    runAlphaForest()
    # runAlphaFrozen()

    # print("Running Alpha Decay Qlearn")
    runAlphaDecayForest()
    # runAlphaDecayFrozen()

    # runEpsDecayForest()
    # runEpsDecayFrozen()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_QLearn_exp()
