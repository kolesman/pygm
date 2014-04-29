import unittest

import numpy as np

import pygm


class testInference(unittest.TestCase):

    def setUp(self):
        # Construct graph. model 1
        f0 = pygm.Factor((10,), np.array([1, 3]))
        f1 = pygm.Factor((20,), np.array([3, 2]))
        f2 = pygm.Factor((30,), np.array([3, 2]))
        f3 = pygm.Factor((40,), np.array([2, 4]))
        f01 = pygm.Factor((10, 20), np.array([1, 3, 1, 2]).reshape(2, 2))
        f12 = pygm.Factor((20, 30), np.array([1, 1, 2, 0]).reshape(2, 2))
        f23 = pygm.Factor((30, 40), np.array([3, 7, 9, 5]).reshape(2, 2))
        f30 = pygm.Factor((10, 40), np.array([2, 1, 2, 0]).reshape(2, 2))
        self.gm1 = pygm.GraphicalModel([f0, f1, f2, f3, f01, f12, f23, f30])
        self.map_state1 = self.gm1.mapBruteForce()
        self.beliefs1 = self.gm1.beliefsBruteForce()

        # Generate random models
        self.random_test_models = []
        for i in range(5):
            random_model = pygm.GraphicalModel.generateRandomGrid(3, 2, 5, submodular=True)
            map_state = random_model.mapBruteForce()
            beliefs = random_model.beliefsBruteForce()
            self.random_test_models.append((random_model, map_state, beliefs))

    def test_mapInference(self):

        exact_sub2_algs = ['Bruteforce', 'AStar', 'GraphCut', 'TrwsExternal', 'QpboExternal']

        for alg in exact_sub2_algs:

            print(alg)

            map_state = self.gm1.getMapState(alg, {})
            self.assertEqual(map_state, self.map_state1)

            for model in self.random_test_models:
                map_state = model[0].getMapState(alg, {})
                self.assertEqual(map_state, model[1])

    def test_probJT(self):
        factor_beliefs = self.gm1.probInference('JTree')
        self.assertTrue(np.all([np.linalg.norm(a1 - a2) < pygm.EPSILON for a1, a2 in zip(self.beliefs1, factor_beliefs)]))

        for model in self.random_test_models:
            factor_beliefs = model[0].probInference('JTree')
            self.assertTrue(np.all([np.linalg.norm(a1 - a2) < pygm.EPSILON for a1, a2 in zip(model[2], factor_beliefs)]))


if __name__ == "__main__":
    unittest.main()
