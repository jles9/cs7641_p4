
import pdb


from hiivemdptoolbox.hiive.mdptoolbox.example import openai

from hiive.mdptoolbox.mdp import PolicyIteration
from hiive.mdptoolbox.example import forest
import matplotlib.pyplot as plt

import gym
from gym.envs.toy_text.frozen_lake import generate_random_map

import time
import random
import numpy as np


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

    plt.style.use('bmh')
    

    plt.figure()
    plt.plot(iterations, mean_vs, label='mean v', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend()
    #plt.show()
    plt.title("Mean V per Iteration for {}".format(problemName))
    plt.savefig('./graphs/PI/{}_meanV.png'.format(problemName))

    plt.figure()
    plt.plot(iterations, max_vs, label='max v', color='green')
    plt.xlabel("Iteration")
    plt.ylabel("Max V")
    plt.legend()
    #plt.show()
    plt.title("Max V per Iteration for {}".format(problemName))
    plt.savefig('./graphs/PI/{}_maxV.png'.format(problemName))

    plt.figure()
    plt.plot(iterations, errors, label='errors', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    #plt.show()
    plt.title("Error per Iteration for {}".format(problemName))
    plt.savefig('./graphs/PI/{}_error.png'.format(problemName))
    
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
    plt.savefig('./graphs/PI/{}_maxV_gamma.png'.format(problemName))

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

        #total_rewards.append(np.sum(rewards)/len(iterations))
        #print(len(iterations))
        sizes.append(size)
        itrs_total.append(len(iterations))

    for size in sizes:
        sizes_as_string.append("{}".format(size))
    
    plt.style.use('bmh')

    plt.figure()
    plt.bar(sizes_as_string, itrs_total)
    plt.xlabel("State Size")
    plt.ylabel("Total Iterations to Converge")
    # plt.legend()
    #plt.show()
    plt.title("Iterations to Converge for {}, Varying State Size".format(problemName))
    plt.savefig('./graphs/PI/{}_size.png'.format(problemName))

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

    pi_agent = PolicyIteration(
        transitions=P,
        reward=R,
        gamma=gamma
    )

    start = time.time()
    result = pi_agent.run()
    runtime = time.time() - start


    cuts = np.sum(pi_agent.policy)
    wait = len(pi_agent.policy) - cuts

    print("Forest Basic Runtime PI: {} \n".format(runtime))
    print(("Forest Basic Policy Cuts PI: {} \n".format(cuts)))
    print(("Forest Basic Policy Waits PI: {} \n".format(wait)))
    plot_basic_exp(result, problemName="forest")


def runFrozenLakeExp():
    # This will be the "large" problem (>200 states)

    # define default hyperparams
    gamma = .998


    # Starting with defaults
    # https://pymdptoolbox.readthedocs.io/en/latest/api/example.html

    random_map = generate_random_map(size=8, p=0.8)

    P, R = openai(
        env_name="FrozenLake-v0",
        render=False,
        desc=random_map
    )

    pi_agent = PolicyIteration(
        transitions=P,
        reward=R,
        gamma=gamma
    )

    start = time.time()
    result = pi_agent.run()
    runtime = time.time() - start

    plot_basic_exp(result, problemName="frozenLake")
    print("Frozen Lake Basic Runtime PI: {} \n".format(runtime))

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
        pi_agent = PolicyIteration(
            transitions=P,
            reward=R,
            gamma=gamma
        )

        result = pi_agent.run()
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
        pi_agent = PolicyIteration(
            transitions=P,
            reward=R,
            gamma=gamma
        )

        result = pi_agent.run()
        results[gamma] = result

    plot_gamma_exp(results,problemName="FrozenLake")

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

        pi_agent = PolicyIteration(
            transitions=P,
            reward=R,
            gamma=gamma, 
        )

        result = pi_agent.run()
        results[size] = result

    plot_sizes(results,problemName="Forest")



def run_PI_exp():
    runForestExp()
    runFrozenLakeExp()
    runGammaForest()
    runGammaFrozen()
    runSizeForest()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run_PI_exp()
