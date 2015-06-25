import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter 
from random import shuffle
import math

ROOTDIR = ""
filepath = []

for dirs, subdir, files in os.walk(ROOTDIR+"data"):
    for ifile in files:
        filepath.append(dirs + "/" + ifile)

#shuffle the filepath so that different classes are read sequentially.
print filepath[0:20]
shuffle(filepath)
print filepath[0:20]

classData = []
seqData = []
countSeq = 0
with open(ROOTDIR+"datafile.csv","w") as writefile:
    for path in filepath:
        #path ='data/22/99.csv'
        s = path.split("/")
        data = []
        with open(path,"r") as filename:		
            count = 0
            countSeq += 1
            for line in filename:
                a,b = line.split(",")
                data.append([a,b[0:-1]])
            i = 2
            #only keep data with sequence length 50
            rng = int(np.floor((len(data)-6)/200)*200)
            #print path, len(data)
            while i<rng:
                dataWin1 = np.array(data[i-1:i+3], dtype=float)
                dataWin2 = np.array(data[i:i+4], dtype=float)
                dataWin3 = np.array(data[i+1:i+5], dtype=float)
               
                vecData = np.array(np.subtract(dataWin2, dataWin1))
                vecData2 = np.array(np.subtract(dataWin3, dataWin2))
               
                accData = np.subtract(vecData2, vecData)
                dirData = np.arctan(np.divide(dataWin2[1],dataWin2[0]))
                
                minVecX, minVecY = np.amin(vecData, axis=0)
                maxVecX, maxVecY = np.amax(vecData, axis=0)
                avgVecX, avgVecY = np.average(vecData, axis=0)
                
                minAccX, minAccY = np.amin(accData, axis=0)
                maxAccX, maxAccY = np.amax(accData, axis=0)
                avgAccX, avgAccY = np.average(accData, axis=0)
                
                minDir = np.amin(dirData, axis=0)
                maxDir = np.amax(dirData, axis=0)
                avgDir = np.average(dirData, axis=0)
                
                temp = [minVecX, minVecY, maxVecX, maxVecY, avgVecX, avgVecY, minDir, maxDir, avgDir, minAccX, minAccY, maxAccX, maxAccY, avgAccX, avgAccY]
                
                #convert all "nan's" and zeros to small values
                for k in range(len(temp)):
                    if math.isnan(temp[k]):
                        temp[k] = 0.00001
                
                writefile.write(str(temp[0])+",")
                writefile.write(str(temp[1])+",")
                writefile.write(str(temp[2])+",")
                writefile.write(str(temp[3])+",")
                writefile.write(str(temp[4])+",")
                writefile.write(str(temp[5])+",")
                writefile.write(str(temp[6])+",")
                writefile.write(str(temp[7])+",")
                writefile.write(str(temp[8])+",")
                writefile.write(str(temp[9])+",")
                writefile.write(str(temp[10])+",")
                writefile.write(str(temp[11])+",")
                writefile.write(str(temp[12])+",")
                writefile.write(str(temp[13])+",")
                writefile.write(str(temp[14])+",")
                writefile.write("\n")
                
                classData.append(int(s[1]))
                count += 1
                if count == 51:
                    countSeq += 1
                    count = 1
                seqData.append(countSeq)
                #print int(s[1]), countSeq, temp
                i += 4

#check the correctness of the created file, with sequence length being fixed
c = Counter(seqData)
seqInfo = c.items()
row = len(seqInfo)
for j in range(row):
    if seqInfo[j][1] != 50:
        print seqInfo[j][1]
        print "Out of Length Sequence"

#since class names are not unique, create a dictionary of names and save it also         
row = len(classData)
c2 = Counter(classData)
list_class = c2.keys()
with open(ROOTDIR+"classfile.csv","w") as filename:
        for j in range(row):		
		filename.write(str(list_class.index(classData[j])))
		filename.write("\n")

with open(ROOTDIR+"classmap.csv","w") as filename:
    for j in range(len(list_class)):
        filename.write(str(list_class[j])+":"+str(j))
        filename.write("\n")
