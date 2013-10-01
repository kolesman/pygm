import sys
import numpy as np

import opengm

import dai


class Factor(object):

    def __init__(self, members, values, probability=False):
        self.__n = len(members)
        self.__members = members

        assert(isinstance(members, tuple))
        assert(isinstance(values, np.ndarray))
        assert(len(values.shape) == len(members))

        values = values.astype('float32')

        if probability:
            illegal_values = values <= 0
            if np.any(illegal_values):
                values[illegal_values] = np.finfo(np.float32).eps
                sys.stderr.write("Warning: illegal probability values(<= 0). These values replaced with machine precision value for float32\n")
            self.__values = -np.log(values)
        else:
            self.__values = values

    @property
    def members(self):
        return self.__members

    @property
    def values(self):
        return self.__values

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

    def __init__(self, factors, normalize_unary_with_pairwise=False):

        self.__factors = factors

        ############################
        new_factors = []
        if normalize_unary_with_pairwise:
            unary_dict = {}
            for factor in factors:
                if len(factor.members) == 1:
                    unary_dict[factor.members[0]] = np.exp(-factor.values)
                    new_factors.append(factor)

            for factor in factors:
                if len(factor.members) == 2:
                    members = factor.members
                    prob_values = np.exp(-factor.values)
                    new_values = (prob_values / unary_dict[members[0]]).T / unary_dict[members[1]]
                    new_factors.append(Factor(members, new_values, probability=True))

            self.__factors = new_factors
        ############################

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

    def getMapState(self, alg, params):
        gm = self._constructOpenGMModel()

        opengm_params = opengm.InfParam(**params)
        inference_alg = getattr(opengm.inference, alg)(gm, parameter=opengm_params)

        inference_alg.infer()
        return inference_alg.arg()

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


def main():
    return 0

if __name__ == "__main__":
    main()
