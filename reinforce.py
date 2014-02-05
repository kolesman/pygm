import lp
import sgd

import numpy as np

from copy import deepcopy

import time


class Environment(object):

    def __init__(self, gms, tol=0.01):
        self.gms = gms
        self.optimal_energies = [lp.solveLPonLocalPolytope(gm)[0] for gm in gms]
        self.tol = tol

    def __enter__(self):
        i = np.random.randint(0, len(self.gms))

        self.gm = deepcopy(self.gms[i])
        update, energy, primal_energy = sgd.computeSubgradient(self.gm)

        grad_norm = np.sum([tree.n_factors for tree in self.gm.tree_decomposition])

        self.state = {'optimal_energy': self.optimal_energies[i],
                      'energy_gap': self.optimal_energies[i] - energy,
                      'grad_norm': grad_norm}

        return self

    def __exit__(self, type, value, traceback):
        self.gm = None
        self.state = None

    def makeAction(self, a):
        update, energy, primal_energy = sgd.computeSubgradient(self.gm)
        sgd.updateDDParams(self.gm, update, a)
        energy = np.sum([tree.Energy(tree.getMapState('DynamicProgramming', {})) for tree in self.gm.tree_decomposition])

        self.state['energy_gap'] = self.state['optimal_energy'] - energy

        reward = -1.0 if np.abs(self.state['energy_gap'] / self.state['optimal_energy']) > self.tol else 0.0

        return deepcopy(self.state), reward

    def getState(self):
        return self.state


class Agent(object):

    def __init__(self, env):
        self.env = env

        self.stateValueParam = [0.0, 0.0, 1.0]
        self.policyParam = [1.0]

    def getAction(self, s):
        return self.policyParam[0] * (s['energy_gap'] / s['grad_norm'])

    def getStateValue(self, s):
        alpha = self.stateValueParam[0]
        beta = self.stateValueParam[1]
        gamma = self.stateValueParam[2]
        return alpha + beta * (s['energy_gap'] + 1) ** gamma

    def getDerivativeStateValue(self, s):
        #alpha = self.stateValueParam[0]
        beta = self.stateValueParam[1]
        gamma = self.stateValueParam[2]

        return 1.0, s['energy_gap'] ** gamma, beta * (s['energy_gap'] + 1) ** gamma * np.log(s['energy_gap'] + 1)

    def evaluatePolicy(self, episodes=10, stepsize=0.01):

        TDlambda = 1.0

        for episode in range(episodes):
            with self.env as env:
                state = env.getState()
                z = np.zeros(len(self.stateValueParam))
                while True:
                    value_initial = self.getStateValue(state)
                    dw = self.getDerivativeStateValue(state)
                    action = self.getAction(state)

                    state, reward = env.makeAction(action)
                    value_new = self.getStateValue(state)

                    delta = reward + value_new - value_initial
                    z = TDlambda * z + dw
                    self.stateValueParam += stepsize * delta * z

                    #print(state)
                    #print(action)
                    #print(value_initial, value_new, reward)
                    #print(dw)
                    #print('---------')
                    print(dw, delta, self.stateValueParam)

                    if reward == 0.0:
                        break
        return 0
