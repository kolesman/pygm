import sys
import numpy as np

from collections import defaultdict, Counter
import time
from copy import deepcopy

import utils

from itertools import product

import opengm
import dai


EPSILON = 10e-8
BIG_INT = 2 ** 32


class Factor(object):

    def __init__(self, members, values, probability=False):
        self.__order = len(members)
        self.__members = members

        values = np.array(values).astype('longdouble')

        assert(isinstance(members, tuple))
        assert(isinstance(values, np.ndarray))
        assert(len(values.shape) == len(members))

        if probability:
            assert(np.all(values >= 0.0 - EPSILON))
            values[values < 0.0] = 0.0
            values = values / np.sum(values)
            values[values > 1.0] = 1.0
            self.__values = -np.log(values)
        else:
            self.__values = values

    @property
    def order(self):
        return self.__order

    @property
    def members(self):
        return self.__members

    @property
    def values(self):
        return self.__values

    @property
    def probabilities(self):
        return np.exp(-self.__values)

    @property
    def cardinalities(self):
        return self.values.shape

    def __str__(self):
        self_str = "Factor with members: %s" % str(self.members) + " and cardinalities: %s" % str(self.cardinalities)
        return self_str


class GraphicalModel(object):

    def __init__(self, factors):

        self.__factors = deepcopy(factors)

        # assert factor uniqness
        members = [factor.members for factor in self.__factors]
        unique_members = list(set(members))
        assert(len(members) == len(unique_members))

        # check numeration. Shoud start from zero and be consecutive
        node_set = set([node for tuples in members for node in tuples])
        assert(list(node_set) == range(len(node_set)))

        # compute and check cardinalities of variables
        self.__cardinalities = np.zeros(len(node_set)).astype('int')
        for factor in self.__factors:
            for member, cardinality in zip(factor.members, factor.cardinalities):
                if self.__cardinalities[member] > 0:
                    assert(self.__cardinalities[member] == cardinality)
                else:
                    self.__cardinalities[member] = cardinality

        self.map_members_index = dict([(factor.members, i) for i, factor in enumerate(self.__factors)])

    @staticmethod
    def generateRandomTree(n, k, sigma, expectation_children=2):

        factor_list = []

        for i in range(n):
            members = (i, )
            values = np.random.normal(0, sigma, k)
            factor_list.append(Factor(members, values))

        edges = utils.generateRandomTree(n, expectation_children)

        for edge in edges:
            members = edge
            values = np.random.normal(0, sigma, (k, k))
            factor_list.append(Factor(members, values))

        return GraphicalModel(factor_list)

    @staticmethod
    def generateRandomGrid(n, k, sigma, submodular=True):
        factor_list = []
        for i in range(n):
            for j in range(n):
                members = (i + j * n, )
                values = np.abs(np.random.normal(0, sigma, k))
                f = Factor(members, values)
                factor_list.append(f)
            for j in range(1, n):
                for members in [(i * n + (j - 1), i * n + j), ((j - 1) * n + i, j * n + i)]:
                    values = np.abs(np.random.normal(0, sigma, (k, k)))
                    if submodular:
                        values[np.diag_indices(k)] = 0.0
                    f = Factor(members, values)
                    factor_list.append(f)
        return GraphicalModel(factor_list)

    @staticmethod
    def loadFromH5(file_name):

        gm = opengm.loadGm(file_name)

        cardinalities = [gm.numberOfLabels(i) for i in range(gm.numberOfVariables)]

        factors = []
        for factor in list(gm.factors()):
            members = tuple(map(int, np.array(factor.variableIndices)))
            values = factor.copyValues().reshape(tuple([cardinalities[member] for member in members])).T
            factors.append(Factor(members, values))

        return GraphicalModel(factors, make_tree_decomposition=True)

    @staticmethod
    def loadFromUAI(file_name):
        f = open(file_name)

        assert(f.next().strip() == "MARKOV")

        n = int(f.next())

        cardinalities = map(int, f.next().strip().split(" "))

        assert(len(cardinalities) == n)

        k = int(f.next().strip())

        members_list = []
        for i in range(k):
            l = map(int, f.next().strip().split(" "))
            order, members = l[0], l[1:]
            assert(order == len(members))
            members_list.append(tuple(members))

        values_list = []
        f.next()
        for i, members in enumerate(members_list):

            shape = tuple([cardinalities[member] for member in members])
            count = int(f.next())
            assert(np.prod(shape) == count)

            values = []
            while True:
                l = f.next().strip()
                if l == "":
                    break
                values += map(float, l.split(" "))

            values = np.array(values).reshape(shape)
            values_list.append(values)

        factors = [Factor(members, values) for members, values in zip(members_list, values_list)]

        for absent in set(range(n)) - set([factor.members[0] for factor in factors if len(factor.members) == 1]):
            factors.append(Factor((absent, ), np.zeros(cardinalities[absent])))

        return GraphicalModel(factors, make_tree_decomposition=True)

    @property
    def n_factors(self):
        return len(self.__factors)

    @property
    def n_vars(self):
        return len(self.__cardinalities)

    @property
    def factors(self):
        return self.__factors

    @property
    def max_order(self):
        return max([factor.order for factor in self.__factors])

    @property
    def cardinalities(self):
        return self.__cardinalities

    @property
    def n_values(self):
        return np.sum([np.prod(factor.values.shape) for factor in self.factors])

    def stateGenerator(self):
        for state in product(*[range(c) for c in self.cardinalities]):
            yield state

    def mapBruteForce(self):

        state_generator = self.stateGenerator()

        best_energy = BIG_INT
        best_state = None

        for state in state_generator:
            energy = self.Energy(state)
            if energy < best_energy:
                best_energy = energy
                best_state = state

        return np.array(best_state)

    def beliefsBruteForce(self):

        prob_table = np.zeros(self.cardinalities)

        for state in self.stateGenerator():
            prob_table[tuple(state)] = np.exp(-self.Energy(state))

        Z = np.sum(prob_table)

        marg_distrs = []

        for marginal_list, member_list in [(marg_distrs, [factor.members for factor in self.factors])]:
            for members in member_list:

                marg_card = [self.cardinalities[member] for member in members]
                marg_distr = np.zeros(marg_card)

                for state in product(*[range(c) for c in marg_card]):

                    index = [slice(None) for dummy in range(len(prob_table.shape))]

                    for member, s in zip(members, state):
                        index[member] = s

                    marg_distr[state] = np.sum(prob_table[tuple(index)]) / Z

                marginal_list.append(marg_distr)

        return marg_distrs

    def _constructOpenGMModel(self):

        openGMModel = opengm.graphicalModel(self.cardinalities, operator="adder")

        for factor in self.factors:
            members = tuple(map(int, list(factor.members)))
            func = openGMModel.addFunction(factor.values)
            openGMModel.addFactor(func, members)

        return openGMModel

    def _constructLibDAIModel(self):

        var_list = []

        for i, cardinality in enumerate(self.cardinalities):
            var_list.append(dai.Var(int(i), int(cardinality)))

        factor_list = []
        for factor in self.factors:
            members = list(factor.members)

            var_set = dai.VarSet(var_list[members[0]])
            for var_number in members[1:]:
                var_set.append(var_list[var_number])
            factor = dai.Factor(var_set)

            factor_list.append(factor)

        for i, (dai_factor, factor) in enumerate(zip(factor_list, self.factors)):
            values = np.exp(-factor.values.T.ravel())
            for j, value in enumerate(values):
                dai_factor[j] = float(value)

        dai_vector_factors = dai.VecFactor()
        [dai_vector_factors.append(dai_factor) for dai_factor in factor_list]

        dai_model = dai.FactorGraph(dai_vector_factors)

        self.dai_factor_list = factor_list
        return dai_model

    def variableList(self):
        variable_set = set(list(sum([factor.members for factor in self.factors], ())))
        return list(variable_set)

    def Energy(self, state):
        state_dict = dict(enumerate(state))

        energy = 0.0
        for factor in self.factors:
            assig = tuple([state_dict[member] for member in factor.members])
            energy += factor.values[assig]

        return energy

    def getMapState(self, alg, params, defaultvalue=0):
        gm = self._constructOpenGMModel()

        opengm_params = opengm.InfParam(**params)
        inference_alg = getattr(opengm.inference, alg)(gm, parameter=opengm_params)

        inference_alg.infer()
        map_state = inference_alg.arg().astype('int')

        return map_state

    def probInference(self, alg, params={}):
        gm = self._constructLibDAIModel()

        parameters = {}
        if alg == 'BP':
            parameters = {'inference': 'SUMPROD', 'updates': 'SEQMAX', 'tol': '1e-6', 'maxiter': '100', 'logdomain': '1'}
        if alg == 'JTree':
            parameters = {'inference': 'SUMPROD', 'updates': 'HUGIN', 'tol': '1e-6'}
        parameters.update(params)

        opts = dai.PropertySet()
        for key, value in parameters.items():
            opts[key] = value

        algorithm = getattr(dai, alg)

        prob_model = algorithm(gm, opts)
        prob_model.init()
        prob_model.run()

        factor_values = []
        for factor, dai_factor in zip(self.factors, self.dai_factor_list):
            belief = prob_model.belief(dai_factor.vars())
            shape = tuple([self.cardinalities[member] for member in factor.members])
            values = np.array([belief[i] for i in range(np.prod(shape))])
            factor_values.append(values.reshape(shape).T)

        return factor_values

    def __call__(self, *args):
        value = getattr(self, args[0])(*args[1:])
        return value


def main():
    return 0

if __name__ == "__main__":
    main()
