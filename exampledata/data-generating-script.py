import numpy as np
#ITEM-ITEM Collaborative filtering data generating script

#users
u = 25
#items
n = 5

#Generate random rating matrix
#M = [[1,3,5,4],[2,1,1,2],[1,3,4,4]]
M = np.random.randint(1,6,size=(u,n))
ofile = open("dataset-U"+str(u)+"-I"+str(n)+".txt",'w')
for i in range(0,u,1):
    for j in range(0,n,1):
        ofile.write(str(j)+","+str(i)+","+str(M[i,j])+"\n")
ofile.close()
N = np.array(M)
#Item correlations
I = np.transpose(N)
C = np.corrcoef(I)

for i in range(0,n,1):
    Z = 0
    V = np.zeros(I.shape[1])
    for j in range(0,n,1):
        if i==j:
            continue
        V = V + C[i,j]*I[j,:]
        Z = Z + abs(C[i,j])
    if i==0:
        PREDICTIONS = V/Z
    else:
        PREDICTIONS = np.vstack((PREDICTIONS,V/Z))
#Predicted values
P1 = np.transpose(PREDICTIONS)
ofile = open("predicted-U"+str(u)+"-I"+str(n)+".txt",'w')
for i in range(0,u,1):
    for j in range(0,n,1):
        ofile.write(str(j)+","+str(i)+","+str(P1[i,j])+"\n")
ofile.close()
print(P1)
