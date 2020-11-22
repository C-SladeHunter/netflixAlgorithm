from collections import defaultdict
from scipy.stats import pearsonr
import numpy as np
import math

#STORE SIMILARITIES IN SIM
SIM = defaultdict(dict)
#STORE RATINGS in ITM
ITM = defaultdict(dict)

#ITM[m][u] stores rating score for movie m and user u
#SIM[m1][m2] stores similarity score between movie m and m1

ifile = open("dataset-U10-I2.txt")
movies = defaultdict(list)
for l in ifile:
    parts = l.strip().split(",")
    ITM[int(parts[0])][int(parts[1])] = float(parts[2])
ifile.close()

for m in ITM.keys():
    for m1 in ITM.keys():
        if m==m1:
            continue
        V = []
        V1 = []
        commonusers = []
        for u in ITM[m].keys():
            if u in ITM[m1]:
                commonusers.append(u)
        for u in commonusers:
            V.append(ITM[m][u])
            V1.append(ITM[m1][u])
        aV = np.array(V)
        aV1 = np.array(V1)
        SIM[m][m1]=pearsonr(aV,aV1)

#STEPS TO DO
#MAKE PREDICTIONS USING ITM AND SIM
#GIVEN a movie m and an user u for which you want to make a prediction do the following
#ITERATE OVER SIM[m] to get all the movies (say m1) similar to m and their similarity weights SIM[m][m1]
#For each movie (m1) similar to m get the rating given by user u for that movie using ITM[m1][u]
#Use the equation (aggregating over all similar movies) to make the prediction
