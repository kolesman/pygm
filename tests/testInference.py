#!/usr/bin/python


import unittest

import numpy as np

import pygm.PGM as PGM
import pygm.utils as utils


np.random.seed(31337)


class testInference(unittest.TestCase):

    def setUp(self):
        # Construct graph. model 1
        f0 = PGM.Factor((10,), np.array([1, 3]))
        f1 = PGM.Factor((20,), np.array([3, 2]))
        f2 = PGM.Factor((30,), np.array([3, 2]))
        f3 = PGM.Factor((40,), np.array([2, 4]))
        f01 = PGM.Factor((10, 20), np.array([1, 3, 1, 2]).reshape(2, 2))
        f12 = PGM.Factor((20, 30), np.array([1, 1, 2, 0]).reshape(2, 2))
        f23 = PGM.Factor((30, 40), np.array([3, 7, 9, 5]).reshape(2, 2))
        f30 = PGM.Factor((10, 40), np.array([2, 1, 2, 0]).reshape(2, 2))
        factors, _ = utils.fixNumeration([f0, f1, f2, f3, f01, f12, f23, f30])
        self.gm1 = PGM.GraphicalModel(factors)
        self.map_state1 = self.gm1.mapBruteForce()
        self.beliefs1 = self.gm1.beliefsBruteForce()

        # High order model
        f0 = PGM.Factor((0, 1, 2), np.array([[[4, 1], [3, 1]], [[4, 3], [3, 4]]]))
        f1 = PGM.Factor((2, 3, 4), np.array([[[2, 5], [2, 5]], [[5, 2], [1, 0]]]))
        f2 = PGM.Factor((1, 2, 3), np.array([[[3, 0], [2, 5]], [[0, 2], [1, 3]]]))
        f2 = PGM.Factor((0, 1, 2, 3), np.array([[[[1, 0], [0, 4]], [[3, 1], [2, 0]]],
                                               [[[2, 5], [1, 2]], [[5, 0], [1, 1]]]]))

        self.gm2 = PGM.GraphicalModel([f0, f1, f2])
        self.map_state2 = self.gm2.mapBruteForce()
        self.beliefs2 = self.gm2.beliefsBruteForce()

        # Generate random models
        self.random_test_models = []
        for i in range(10):
            random_model = PGM.GraphicalModel.generateRandomGrid(3, 2, 10, submodular=True)
            map_state = random_model.mapBruteForce()
            beliefs = random_model.beliefsBruteForce()
            self.random_test_models.append((random_model, map_state, beliefs))

    def test_mapInference(self):

        exact_algs = ['Bruteforce', 'AStar']
        exact_sub2_algs = ['Bruteforce', 'AStar', 'GraphCut', 'TrwsExternal', 'QpboExternal']

        for alg in exact_sub2_algs:

            map_state = self.gm1.getMapState(alg, {})
            self.assertTrue(np.all(map_state == self.map_state1))

            for model in self.random_test_models:
                map_state = model[0].getMapState(alg, {})
                self.assertTrue(np.all(map_state == model[1]))

        for alg in exact_algs:

            map_state = self.gm2.getMapState(alg, {})
            self.assertTrue(np.all(map_state == self.map_state2))

    def test_probJT(self):

        exact_algs = ['JTree']

        for alg in exact_algs:

            factor_beliefs = self.gm1.probInference(alg)
            self.assertTrue(np.all([np.linalg.norm(a1 - a2) < PGM.EPSILON for a1, a2 in zip(self.beliefs1, factor_beliefs)]))

            factor_beliefs = self.gm2.probInference(alg)
            self.assertTrue(np.all([np.linalg.norm(a1 - a2) < PGM.EPSILON for a1, a2 in zip(self.beliefs2, factor_beliefs)]))

            for model in self.random_test_models:
                factor_beliefs = model[0].probInference(alg)
                self.assertTrue(np.all([np.linalg.norm(a1 - a2) < PGM.EPSILON for a1, a2 in zip(model[2], factor_beliefs)]))


if __name__ == "__main__":
    unittest.main()
