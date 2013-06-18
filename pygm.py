import sys
import numpy as np

import opengm


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
                sys.stderr.write("Warning: illegal probability values(<= 0). These values replaced with machine precision for float32\n")
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

    def __init__(self, factors):
        #assert(all([isinstance(factor, Factor) for factor in factors]))
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

    def getMapState(self, alg, params):
        gm = self._constructOpenGMModel()

        opengm_params = opengm.InfParam(**params)
        inference_alg = getattr(opengm.inference, alg)(gm, parameter=opengm_params)

        inference_alg.infer()
        return inference_alg.arg()


def main():
    return 0

if __name__ == "__main__":
    main()
