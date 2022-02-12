from src.tests.test_db import BenchMarkTest
print("starting")
t = BenchMarkTest()
t.setUp()
t.test_spill()
#t.test_spill()
print("FINISHED SPILLING FROM DB")
