import unittest

import numpy as np

import pygm


class testInference(unittest.TestCase):

    def setUp(self):
        # Construct graph. model 1
        f0 = pygm.Factor((10,), np.array([0.8, 0.2]), probability=True)
        f1 = pygm.Factor((20,), np.array([0.7, 0.3]), probability=True)
        f2 = pygm.Factor((30,), np.array([0.5, 0.5]), probability=True)
        f3 = pygm.Factor((40,), np.array([0.6, 0.4]), probability=True)
        f01 = pygm.Factor((10, 20), np.array([0.4, 0.1, 0.1, 0.4]).reshape(2, 2), probability=True)
        f12 = pygm.Factor((20, 30), np.array([0.1, 0.4, 0.4, 0.1]).reshape(2, 2), probability=True)
        f23 = pygm.Factor((30, 40), np.array([0.4, 0.1, 0.1, 0.4]).reshape(2, 2), probability=True)
        f30 = pygm.Factor((10, 40), np.array([0.1, 0.4, 0.4, 0.1]).reshape(2, 2), probability=True)
        self.gm1 = pygm.GraphicalModel([f0, f1, f2, f3, f01, f12, f23, f30])
        self.map_state1 = self.gm1.mapBruteForce()

    def test_mapInference(self):
        map_state = self.gm1.getMapState('TrwsExternal', {})
        self.assertEqual(map_state, self.map_state1)

#    def test_probJT(self):
#        factor_beliefs = self.gm1.probInference('JTree')
#        self.assertTrue(np.all(map(lambda (a1, a2): np.all(a1 - a2 < 10e-6), zip(factor_beliefs, self.beliefs1))))


if __name__ == "__main__":
    unittest.main()
