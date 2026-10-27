import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from generators import sample, Settings
from benchmark import quality, synthesize
class BenchmarkTests(unittest.TestCase):
 def test_smote_exact_budget_and_finite(self):
  x,y=sample(600,13,'s1'); n=int(sum(y==0)-sum(y==1))
  gen,_=synthesize('smote',x,y,n,13,'s1',Settings(),100)
  self.assertEqual(gen.shape,(n,6));self.assertTrue(np.isfinite(gen).all())
 def test_oracle_budget(self):
  x,y=sample(600,13,'s1');n=int(sum(y==0)-sum(y==1))
  gen,_=synthesize('oracle',x,y,n,13,'s1',Settings(),100)
  self.assertEqual(gen.shape,(n,6))
 def test_quality_finite(self):
  x,y=sample(500,2,'s1');ref,ry=sample(4000,3,'s1')
  q=quality(x[y==1],ref[ry==1],x[y==1],StandardScaler().fit(x))
  self.assertTrue(all(np.isfinite(v) for v in q.values()))
if __name__=='__main__':unittest.main()
