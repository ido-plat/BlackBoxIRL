from src.tests.fake_agent_test_eval import FakeAgentTestEval
from src.config import Config


def _assertions():
    assert Config.airl_num_transitions >= int(1e6) and Config.in_lab
# _assertions()
print('starting testing')
t = FakeAgentTestEval()
t.setUp()
print('made test init')
# t.test_full_pipeline()
t.test_mean_fake_score()
print('done')
