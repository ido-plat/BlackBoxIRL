from src.tests.benchmarking_test import BenchMarkTest
from src.config import Config


def _assertions():
    assert Config.airl_num_transitions >= int(1e6) and Config.in_lab
# _assertions()
print('starting testing')
t = BenchMarkTest()
t.setUp()
print('made test init')
# t.test_full_pipeline()
t.test_partial_pipelie()
print('done')
