from src.tests.benchmarking_test import BenchMarkTest
print('starting testing')
t = BenchMarkTest()
t.setUp()
print('made test init')
t.test_full_pipeline()