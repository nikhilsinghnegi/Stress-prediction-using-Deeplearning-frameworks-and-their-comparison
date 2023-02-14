# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__

import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

data_set = 120

for i in range(data_set):
    j = str(i+1)
    input_name = 'New_Master_von_' + j
    input_file_name = 'D:\\8th sem\\BTP_II\\DiNN-framework-Stress-Prediction-for-Heterogeneous-Media-master\\Data processing\\Plate with circular cutout model\\check_here2\\' + input_name + '.inp'
    mdb.JobFromInputFile(name=input_name,
        inputFileName=input_file_name, type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
        userSubroutine='', scratch='', resultsFormat=ODB,
        multiprocessingMode=DEFAULT, numCpus=1, numGPUs=1)
    mdb.jobs[input_name].submit(consistencyChecking=OFF)

session.mdbData.summary()