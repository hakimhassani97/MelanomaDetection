import time
import numpy as np
from NashEnumSupport import NashEnumSupport
from NashEnumSommet import NashEnumSommet
from NashLemkHowson import NashLemkHowson
import nashpy

for i in range(25,26,1):
    A = np.int8(10 * np.random.random((10, 10)))
    B = np.int8(10 * np.random.random((10, 10)))
    t1 = time.time()
    # J1 = NashEnumSupport(A,B).EQ()
    #J1 = NashEnumSommet(A,B).EQ()
    J1 = NashLemkHowson(A,B).EQs()
    #J1 = NashLemkHowson(A,B).EQ()
    t2 = time.time()
    print (J1)
    print(t2-t1)