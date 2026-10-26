import unittest
import numpy as np
from generators import sample, SCENARIOS

class GeneratorTests(unittest.TestCase):
    def test_shapes_finite_labels(self):
        for s in SCENARIOS:
            x,y=sample(1000,42,s)
            self.assertEqual(x.shape,(1000,6))
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(set(np.unique(y)).issubset({0,1}))
    def test_repeatability(self):
        for s in SCENARIOS:
            a,b=sample(500,42,s)
            c,d=sample(500,42,s)
            np.testing.assert_array_equal(a,c)
            np.testing.assert_array_equal(b,d)
    def test_split_independence(self):
        a,_=sample(500,42,'s1','train')
        b,_=sample(500,42,'s1','test')
        self.assertFalse(np.array_equal(a,b))
    def test_s1_s3_train_control(self):
        a,y=sample(500,42,'s1','train')
        for s in ('s3a','s3b'):
            b,z=sample(500,42,s,'train')
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(y,z)
    def test_shift_changes_test_not_labels(self):
        x,y=sample(2000,42,'s1','test')
        for s in ('s3a','s3b'):
            z,t=sample(2000,42,s,'test')
            np.testing.assert_array_equal(y,t)
            self.assertFalse(np.array_equal(x,z))
    def test_s2_difficulty_parameter(self):
        a,y=sample(1000,42,'s2a')
        b,z=sample(1000,42,'s2b')
        np.testing.assert_array_equal(y,z)
        self.assertFalse(np.array_equal(a,b))

if __name__=='__main__': unittest.main()
