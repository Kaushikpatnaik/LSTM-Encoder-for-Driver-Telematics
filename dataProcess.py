import numpy as np
import os
import random
from collections import defaultdict, Counter
from sklearn.utils import resample, shuffle
import math

def createFeatures(dataWin1, dataWin2, dataWin3):
    # given three raw data windows compute velocity accelaration
    # and change in direction
    
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
    
    featVector = [minVecX, minVecY, maxVecX, maxVecY, avgVecX, avgVecY, minDir, maxDir, avgDir, minAccX, minAccY, maxAccX, maxAccY, avgAccX, avgAccY]
    
    return featVector


def getData(allpath):
    # function which given a filepath returns seqID a list of sequence IDs
    # and dataFile a numpy array containing the features
    
    dataFile = []
    seqID = []
    filepath = []
    
    for dirs, subdir, files in os.walk(allpath):
        for ifile in files:
            filepath.append(dirs + "/" + ifile)

    for path in filepath:
        s = path.split("/")
        data = []
        with open(path,"r") as filename:		
            count = 0
            countSeq = 1
            temp_collec = []
            for line in filename:
                a,b = line.split(",")
                data.append([a,b[0:-1]])
            i = 2
            #round off the trip length to the nearest 200
            rng = int(np.floor((len(data)-6)/200)*200)
            while i<rng:
                dataWin1 = np.array(data[i-1:i+3], dtype=float)
                dataWin2 = np.array(data[i:i+4], dtype=float)
                dataWin3 = np.array(data[i+1:i+5], dtype=float)
               
                temp = createFeatures(dataWin1, dataWin2, dataWin3)
                
                #convert all "nan's" and zeros to small values
                for k in range(len(temp)):
                    if math.isnan(temp[k]):
                        temp[k] = 0.00001                
                    temp_collec.append(temp[k])
                
                count += 1
                if count == 50:
                    #print len(temp_collec)
                    dataFile.append(temp_collec)
                    temp = s[3].split(".")
                    seqID.append(s[2]+"-"+temp[0]+"-"+str(countSeq))
                    temp_collec = []
                    countSeq += 1
                    count = 0
                i += 4
        
    dataFile = np.array(dataFile)
    seqID = np.array(seqID)
    returnVal = [seqID, dataFile]
    
    return returnVal
    

if __name__ == "__main__":
    
    ROOTDIR = "./"
    subdirpath = []
    
    # read through the directory to obtain subdirectory for each driver
    for dirs, subdirs, files in os.walk(ROOTDIR+"data"):
        for subdir in subdirs:
            subdirpath.append(dirs+"/"+subdir)
    
    # for each driver, we collect data from 40 other drivers as false
    # trips
    driver_collec = defaultdict(list)
    for subdir in subdirpath:
        s = subdir.split('/')
        driver_collec[s[2]].append(subdir)
        for j in range(1):
            #draw a random choice
            temp = random.choice(subdirpath)
            if temp != subdir:
                driver_collec[s[2]].append(temp)
    
    # for each key of the dictionary we generate a csv file
    for key in driver_collec.keys():
        filepath = []
        values = driver_collec[key]
        print "Creating file for driver: " + str(key)
        # get data for the driver
        [dSeqID, dData] = getData(values[0])
        
        # get data for other drivers
        [oSeqID, oData] = getData(values[1])
        
        '''
        k = 2
        while k < len(values[2:]):
            [temp1, temp2] = getData(values[k])
            #print temp1.shape, temp2.shape
            #print oSeqID.shape, oData.shape
            oSeqID = np.hstack((oSeqID, temp1))
            oData =  np.vstack((oData, temp2))
            k += 1
        '''
        
        
        print oData.shape, dData.shape
        
        print "Resampling Data"
        
        if oData.shape[0] > dData.shape[0]:
            row = dData.shape[0]
            trow = oData.shape[0]
            # resample data with replacement
            while row < (trow-row):
                temp1, temp2 = resample(dData, dSeqID, n_samples = row, random_state = 0)
                #print temp1.shape, temp2.shape
                #print dSeqID.shape, dData.shape
                dSeqID = np.hstack((dSeqID, temp2))
                dData =  np.vstack((dData, temp1))
                row += row
            
            diff = trow - row
            temp1, temp2 = resample(dData, dSeqID, n_samples = diff, random_state = 0)
            dSeqID = np.hstack((dSeqID, temp2))
            dData =  np.vstack((dData, temp1))
        else:
            row = oData.shape[0]
            trow = dData.shape[0]
            # resample data with replacement
            while row < (trow-row):
                temp1, temp2 = resample(oData, oSeqID, n_samples = row, random_state = 0)
                #print temp1.shape, temp2.shape
                #print dSeqID.shape, dData.shape
                oSeqID = np.hstack((oSeqID, temp2))
                oData =  np.vstack((oData, temp1))
                row += row
            
            diff = trow - row
            temp1, temp2 = resample(oData, oSeqID, n_samples = diff, random_state = 0)
            oSeqID = np.hstack((oSeqID, temp2))
            oData =  np.vstack((oData, temp1))
        
        print oData.shape, dData.shape
        print dSeqID.shape, oSeqID.shape
        
        # append data
        seqID = np.hstack((dSeqID, oSeqID))
        data =  np.vstack((dData, oData))
        
        print "Shuffling Data"
        
        # shuffle
        seqID, data = shuffle(seqID, data, random_state = 0)
        row, col = data.shape
        
        print "Created Dataset in desired format"
        
        # write to file
        with open(ROOTDIR+"proc_data/datafile_"+str(key)+".csv","w") as filename:
            for i in range(row):
                writedata = data[i]
                newwritedata = np.reshape(writedata, (50,15))
                for j in range(50):
                    for k in range(14):
                        filename.write(str(newwritedata[j][k]))
                        filename.write(",")
                    filename.write(str(newwritedata[j][14]))
                    filename.write("\n")
                    

        # since class names are not unique, create a dictionary of names and save it also         
        with open(ROOTDIR+"proc_data/classfile_"+str(key)+".csv","w") as filename:
            for i in range(row):
                temp = seqID[i].split("-")
                #print temp[0], str(key), temp[0] == str(key) 
                for k in range(50):
                    writedata = temp[0]
                    if writedata == str(key):
                        filename.write(str(1))
                    else:
                        filename.write(str(2))		
                    filename.write("\n")

        # write out the mapping
        with open(ROOTDIR+"proc_data/classmap_"+str(key)+".csv","w") as filename:
            for i in range(row):
                writedata = seqID[i]
                filename.write(writedata)
                filename.write("\n")
