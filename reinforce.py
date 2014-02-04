import lp
import sgd

import numpy as np

from copy import deepcopy


class Environment:

    def __init__(self, gms, tol=0.01):
        self.gms = gms
        self.optimal_energies = [lp.solveLPonLocalPolytope(gm)[0] for gm in gms]
        self.tol = tol

    def __enter__(self):
        i = np.random.random(len(self.gms))

        self.gm = deepcopy(self.gms[i])
        update, energy, primal_energy = sgd.computeSubgradient(self.gm)

        grad_norm = np.sum(for tree in self.gm.tree_decomposition)

        self.optimal = self.optimal_energies[i]
        self.state = {'energy_gap': self.optimal - energy, }

    def __exit__(self, type, value, traceback):
        self.gm = None
        self.state = None

    def makeAction(self, a):
        update, energy, primal_energy = sgd.computeSubgradient(self.gm)
        sgd.updateDDParams(self.gm, update, a)
        self.state = self.optimal - np.sum([tree.Energy(tree.getMapState('DynamicProgramming')) for tree in self.gm.tree_decomposition])

        reward = -1.0 if (self.optimal_energy - energy) > self.tol else 0.0

        return self.state, reward

    def getState(self):
        return self.state


class Agent:

    def __init__(self, env):
        self.env = env

        self.stateValueParam = [1.0, 0.5]
        self.policyParam = [1.0]

    def getAction(s):
        return

    def evaluatePolicy(self, episodes=10, stepsize=0.01):

        for episode in range(episodes):
            with self.env as env:
