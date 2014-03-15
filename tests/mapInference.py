import unittest

import numpy as np

import pygm


class testInference(unittest.TestCase):

    def setUp(self):
        # Construct graph. model 1
        f0 = pygm.Factor((10,), np.array([0.5, 0.5]), probability=True)
        f1 = pygm.Factor((11,), np.array([0.7, 0.3]), probability=True)
        f2 = pygm.Factor((12,), np.array([0.5, 0.5]), probability=True)
        f3 = pygm.Factor((13,), np.array([0.1, 0.9]), probability=True)
        f01 = pygm.Factor((10, 11), np.array([0.3, 0.2, 0.2, 0.3]).reshape(2, 2), probability=True)
        f12 = pygm.Factor((11, 12), np.array([0.3, 0.2, 0.2, 0.3]).reshape(2, 2), probability=True)
        f23 = pygm.Factor((12, 13), np.array([0.4, 0.1, 0.1, 0.4]).reshape(2, 2), probability=True)
        f30 = pygm.Factor((10, 13), np.array([0.4, 0.1, 0.1, 0.4]).reshape(2, 2), probability=True)
        self.gm1 = pygm.GraphicalModel([f0, f1, f2, f3, f01, f12, f23, f30])
        self.map_state1 = [1, 0, 1, 1]
        self.beliefs1 = [np.array([0.29180886, 0.70819114]),
                         np.array([0.61399318, 0.38600682]),
                         np.array([0.29180886, 0.70819114]),
                         np.array([0.11843003, 0.88156997]),
                         np.array([[0.22218429, 0.39180888], [0.06962457, 0.31638225]]),
                         np.array([[0.22218429, 0.06962457], [0.39180888, 0.31638225]]),
                         np.array([[0.09829352, 0.02013652], [0.19351535, 0.68805462]]),
                         np.array([[0.09829352, 0.02013652], [0.19351535, 0.68805462]])]

    def test_mapInference(self):
        map_state = self.gm1.getMapState('Bruteforce', {})
        self.assertEqual(list(map_state), self.map_state1)

#    def test_probJT(self):
#        factor_beliefs = self.gm1.probInference('JTree')
#        self.assertTrue(np.all(map(lambda (a1, a2): np.all(a1 - a2 < 10e-6), zip(factor_beliefs, self.beliefs1))))


if __name__ == "__main__":
    unittest.main()
