import sys
import numpy as np

import opengm

import dai

from compiler.ast import flatten

from collections import defaultdict
from collections import Counter

import time

from copy import deepcopy

import pygm.utils as utils


class Factor(object):

    def __init__(self, members, values, probability=False):
        self.__n = len(members)
        self.__members = members

        values = np.array(values).astype('longdouble')

        assert(isinstance(members, tuple))
        assert(isinstance(values, np.ndarray))
        assert(len(values.shape) == len(members))

        if probability:
            illegal_values = values <= 0
            if np.any(illegal_values):
                values[illegal_values] = 10e-9
                sys.stderr.write("Warning: illegal probability values(<= 0). These values replaced with machine precision value for float32\n")
            values = values / np.sum(values)
            self.__values = -np.log(values)
        else:
            self.__values = values

    @property
    def members(self):
        return self.__members

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, value):
        self.__values = value

    @property
    def probs(self):
        return np.exp(-self.__values)

    @property
    def order(self):
        return self.__n

    @property
    def cardinalities(self):
        return self.values.shape

    def __str__(self):
        members_str = "Members: " + (self.n * "%i ").rstrip(" ") % self.members
        return members_str


class GraphicalModel(object):

    def __init__(self, factors, make_tree_decomposition=False, normalize_unary_with_pairwise=(False, None)):

        self.__factors = factors

        # compute variable cardinalities
        cardinalities_dict = {}
        for factor in self.__factors:
            for member, cardinality in zip(factor.members, factor.cardinalities):
                if member in cardinalities_dict:
                    if cardinalities_dict[member] != cardinality:
                        sys.stderr.write("Error: variable %i has inconsistent cardinalities\n" % member)
                        sys.exit(1)
                else:
                    cardinalities_dict[member] = cardinality

        self.__cardinalities = [1] * (max(cardinalities_dict.keys()) + 1)

        for variable, cardinality in cardinalities_dict.items():
            self.__cardinalities[variable] = cardinality

        self.map_members_index = dict([(factor.members, i) for i, factor in enumerate(self.__factors)])

        if make_tree_decomposition:
            self.tree_decomposition = self._treeDecomposition()
            self.tree_decomposition_edge_mask = self._treeDecompositionEdgeMask()

    @staticmethod
    def generateRandomGrid(n, k, sigma, d_max, bias0=0.0, make_tree_decomposition=True):
        factor_list = []
        for i in range(n):
            for j in range(n):
                members = (i + j * n, )
                #values = np.sort(np.random.normal(0, sigma, k))
                values = np.random.normal(0, sigma, k)
                values[0] -= bias0
                f = Factor(members, values)
                factor_list.append(f)
            for j in range(1, n):
                for members in [(i * n + (j - 1), i * n + j), ((j - 1) * n + i, j * n + i)]:
                    values = d_max * np.abs(np.random.normal(0, sigma, (k, k)))
                    values[0][0] -= bias0
                    values[np.diag_indices(k)] = 0.0
                    f = Factor(members, values)
                    factor_list.append(f)
        return GraphicalModel(factor_list, make_tree_decomposition=make_tree_decomposition)

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
    def average_element(self):
        return np.average(np.hstack([np.abs(factor.values).ravel() for factor in self.factors]))

    @property
    def n_values(self):
        return np.sum([np.prod(factor.values.shape) for factor in self.factors])

    def _treeDecomposition(self):

        if self.max_order > 2:
            raise NotImplemented

        edges = [factor.members for factor in self.factors if len(factor.members) == 2]

        unary_factors = dict([(factor.members[0], factor) for factor in self.factors if len(factor.members) == 1])
        pair_factors = dict([(factor.members, factor) for factor in self.factors if len(factor.members) == 2])

        subtrees = utils.decomposeOnTrees(edges)

        edge_count = Counter([edge for tree in subtrees for edge in tree])
        node_count = Counter([node for tree in subtrees for node in utils.listNodes(tree)])

        decompositions = []

        for tree in subtrees:
            current_decomposition = []
            current_unary = set()
            for edge in tree:
                pair_factor = deepcopy(pair_factors[edge])
                pair_factor.values = pair_factor.values / edge_count[edge]
                current_decomposition.append(pair_factor)

                if edge[0] not in current_unary:
                    unary_factor = deepcopy(unary_factors[edge[0]])
                    unary_factor.values = unary_factor.values / node_count[edge[0]]
                    current_decomposition.append(unary_factor)

                if edge[1] not in current_unary:
                    unary_factor = deepcopy(unary_factors[edge[1]])
                    unary_factor.values = unary_factor.values / node_count[edge[1]]
                    current_decomposition.append(unary_factor)

                current_unary.add(edge[0])
                current_unary.add(edge[1])

            decompositions.append(GraphicalModel(current_decomposition))

        return decompositions

    def _treeDecompositionEdgeMask(self):

        decomposition_edge_sets = [set([factor.members for factor in tree.factors if len(factor.members) == 2])
                                   for tree in self.tree_decomposition]

        edge_mask = {}

        edges = [factor.members for factor in self.factors if len(factor.members) == 2]

        for i, j in edges:
            edge_mask[(i, j)] = np.array([(i, j) in s for s in decomposition_edge_sets]).astype('bool')

        return edge_mask

    def _constructOpenGMModel(self):
        openGMModel = opengm.graphicalModel(self.cardinalities, operator="adder")

        for factor in self.factors:
            func = openGMModel.addFunction(factor.values)
            openGMModel.addFactor(func, factor.members)

        return openGMModel

    def _constructLibDAIModel(self):

        var_list = []

        for i, cardinality in enumerate(self.cardinalities):
            var_list.append(dai.Var(i, cardinality))

        factor_list = []
        for factor in self.factors:
            factor = dai.Factor(dai.VarSet(*[var_list[i] for i in factor.members]))
            factor_list.append(factor)

        for i, (dai_factor, factor) in enumerate(zip(factor_list, self.factors)):
            values = factor.values.ravel()
            for j, value in enumerate(values):
                dai_factor[j] = float(np.exp(-value))

        dai_vector_factors = dai.VecFactor()
        [dai_vector_factors.append(dai_factor) for dai_factor in factor_list]

        dai_model = dai.FactorGraph(dai_vector_factors)

        self.dai_factor_list = factor_list
        return dai_model

    def _insertObservation(self, observation, partial):
        if self.max_order > 2:
            raise NotImplemented

        observation_dict = dict([(i, single) for i, (take, single) in enumerate(zip(partial, observation)) if take])

        new_factors = []
        for factor in self.__factors:
            members = factor.members
            if len(members) == 1:
                if members[0] in observation_dict:
                    new_factor = Factor(members, [1.0], probability=True)
                else:
                    new_factor = factor
            if len(members) == 2:
                if members[0] in observation_dict and members[1] in observation_dict:
                    new_factor = Factor(members, [[1.0]], probability=True)
                elif members[0] in observation_dict:
                    obs = observation_dict[members[0]]
                    new_factor = Factor(members, [factor.probs[obs]], probability=True)
                elif members[1] in observation_dict:
                    obs = observation_dict[members[1]]
                    new_factor = Factor(members, np.array([factor.probs[:, obs]]).T, probability=True)
                else:
                    new_factor = factor
            new_factors.append(new_factor)
        return GraphicalModel(new_factors)

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

        if alg == 'Mqpbo':
            partial = inference_alg.partialOptimality()
            new_fg = self._insertObservation(map_state, partial)
            comp_map_state = new_fg.getMapState('TrwsExternal', {'steps': 10})
            map_state[~partial] = comp_map_state[~partial]

        variable_list = self.variableList()
        variable_mask = np.zeros(self.n_vars).astype('bool')
        variable_mask[variable_list] = True

        map_state[~variable_mask] = defaultvalue

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
            factor_values.append(values.reshape(shape))

        return factor_values

    def __call__(self, *args):
        value = getattr(self, args[0])(*args[1:])
        return value


def main():
    return 0

if __name__ == "__main__":
    main()
