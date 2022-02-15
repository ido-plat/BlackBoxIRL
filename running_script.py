from src.tests.benchmarking_test import BenchMarkTest
from src.config import print_to_cfg_log
print_to_cfg_log('starting testing')
t = BenchMarkTest()
t.setUp()
print_to_cfg_log('made test init')
# t.test_single_classification()
# t.test_full_pipeline()
# t.test_compare_expart_agent_noise()
t.test_full_pipeline()
# t.test_partial_pipeline()
print_to_cfg_log('done')
