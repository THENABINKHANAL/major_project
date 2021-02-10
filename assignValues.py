import numpy as np
def assignValues(a):
    b=np.transpose(a)
    output=[]
    output2=[]
    output1=[]
    minValues=[]
    outputAdded=0
    for i in range(len(b)):
        output.append(-1)
        minValues.append([i,b[i].min()])
    minLength=min(len(b),len(b[0]))
    while outputAdded<minLength:
        minValues = sorted(minValues, key=lambda a_entry: a_entry[1])
        j=0;
        for k in range(len(minValues)):
            minpos=np.argmin(b[minValues[j][0]])
            if(minpos in output):
                b[minValues[j][0]][minpos]=100
                minValues[j][1]=b[minValues[j][0]].min()
                break;
            outputAdded+=1
            output[minValues[j][0]]=minpos
            minValues.pop(j)
    for i in range(len(output)):
        if(output[i]==-1):
            continue
        output1.append(i)
        output2.append(output[i])
    return output2,output1