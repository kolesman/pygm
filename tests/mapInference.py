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
        f01 = pygm.Factor((10, 20), np.array([1, 3, 1, 4]).reshape(2, 2))
        f12 = pygm.Factor((20, 30), np.array([5, 1, 2, 3]).reshape(2, 2))
        f23 = pygm.Factor((30, 40), np.array([3, 2, 1, 5]).reshape(2, 2))
        f30 = pygm.Factor((10, 40), np.array([1, 1, 2, 3]).reshape(2, 2))
        self.gm1 = pygm.GraphicalModel([f0, f1, f2, f3, f01, f12, f23, f30])
        self.map_state1 = self.gm1.mapBruteForce()
        self.beliefs1 = self.gm1.probabilityInferenceBruteForce()

    def test_mapInference(self):
        map_state = self.gm1.getMapState('TrwsExternal', {})
        print(map_state, self.map_state1)
        self.assertEqual(map_state, self.map_state1)

    def test_probJT(self):
        factor_beliefs = self.gm1.probInference('JTree')
        print(self.beliefs1)
        print(factor_beliefs)
        print([np.linalg.norm(a1 - a2) < pygm.EPSILON for a1, a2 in zip(self.beliefs1, factor_beliefs)])
        self.assertTrue(np.all([np.linalg.norm(a1 - a2) < pygm.EPSILON for a1, a2 in zip(self.beliefs1, factor_beliefs)]))


if __name__ == "__main__":
    unittest.main()
