from src.tests.benchmarking_test import BenchMarkTest
print('starting testing')
t = BenchMarkTest()
t.setUp()
print('made test init')
t.test_single_classification()
# t.test_partial_pipeline()
# t.test_compare_expart_agent_noise()
print('done')
