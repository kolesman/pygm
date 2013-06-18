import unittest

import numpy as np

import pygm


class testMapInference(unittest.TestCase):

    def setUp(self):
        # Construct graph. model 1
        f0 = pygm.Factor((0,), np.array([0.1, 0.9]), probability=True)
        f1 = pygm.Factor((1,), np.array([0.2, 0.8]), probability=True)
        f2 = pygm.Factor((2,), np.array([0.3, 0.7]), probability=True)
        f3 = pygm.Factor((3,), np.array([0.2, 0.8]), probability=True)
        f01 = pygm.Factor((0, 1), np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2), probability=True)
        f12 = pygm.Factor((1, 2), np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2), probability=True)
        f23 = pygm.Factor((1, 2), np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2), probability=True)
        f30 = pygm.Factor((1, 2), np.array([0.5, 0.5, 0.5, 0.5]).reshape(2, 2), probability=True)
        self.gm1 = pygm.GraphicalModel([f0, f1, f2, f3, f01, f12, f23, f30])
        self.map_state1 = [1, 1, 1, 1]

    def test_mapInference(self):
        map_state = self.gm1.getMapState('Bruteforce', {})
        self.assertEqual(list(map_state), self.map_state1)


if __name__ == "__main__":
    unittest.main()
