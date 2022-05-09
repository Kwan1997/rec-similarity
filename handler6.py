from pcc import *
import numpy as np
from cdsds import CalMetric
import time
import timeit


resultsDict = {}
nppnsp = {}
neighbours0 = [10, 20, 30, 40, 50, 60]
neighbours = [20, 40, 60, 80, 100, 120]
neighbours2 = [40, 80, 120, 160, 200, 240]
neighbours3 = [10, 40, 70, 100, 130, 160]
neighbours53 = [20, 40, 60, 80, 100, 120, 140, 180, 220, 260, 300, 340, 400, 460, 520, 580, 640, 700]
neighbourst1 = [10, 40, 70, 100, 130, 160]

start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(PCC, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is PCC model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./pcc.npy', resultsDict)
# np.save('./pcc2.npy', nppnsp)
np.save('./pcc_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is PCC model')
print('#' * 100)

resultsDict = {}
nppnsp = {}
start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(CPCC, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is CPCC model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./cpcc.npy', resultsDict)
# np.save('./cpcc2.npy', nppnsp)
np.save('./cpcc_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is CPCC model')
print('#' * 100)

resultsDict = {}
nppnsp = {}
start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(JMSD, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is JMSD model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./jmsd.npy', resultsDict)
# np.save('./jmsd2.npy', nppnsp)
np.save('./jmsd_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is JMSD model')
print('#' * 100)

resultsDict = {}
nppnsp = {}
start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(Cosine, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is cosine model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./cosine.npy', resultsDict)
# np.save('./cosine2.npy', nppnsp)
np.save('./cosine_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is cosine model')
print('#' * 100)

resultsDict = {}
nppnsp = {}
start = timeit.default_timer()
resultsDict['mae'], resultsDict['rmse'], resultsDict['pre'], resultsDict['rec'], resultsDict['f1'], nppnsp['npp'], \
nppnsp['nsp'] = CalMetric().Curvecvcalculate(ACPCC, fold=5, neighbours=neighbourst1)
end = timeit.default_timer()
time_consumed = end - start
print('#' * 100)
print('This is ACPCC model')
print('#' * 100)
for key, val in resultsDict.items():
    print(key, val)
print('Saving dictionary to memory......')
# np.save('./acpcc.npy', resultsDict)
# np.save('./acpcc2.npy', nppnsp)
np.save('./acpcc_time.npy', time_consumed)
print(time_consumed)
print('Saving dictionary to memory successfully!')
print('#' * 100)
print('This is ACPCC model')
print('#' * 100)
