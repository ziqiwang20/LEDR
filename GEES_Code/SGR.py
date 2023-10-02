import numpy as np
import pandas as pd
# from numba import njit
from scipy.spatial import distance # Euclidean distance
from utils import adjacent_servers, dominant_resource_demand
import timeit
# search the minimum privacy/resource ratio in the neighbor servers,
# and return the number of the server with resource availability guarentee

# @jit(nopython=True)
def Secure_Greedy_Response(users, servers,obfu, bia, w_2, w_3):
    adj = adjacent_servers(users, servers) # get adjacent server set
    allocation = pd.DataFrame(np.zeros((len(users),len(servers)))) # initialize allocation matrix
    users = dominant_resource_demand(users, servers, adj) # sort users demand in DRD
    servers1 = servers
    # begin allocation:
    # get adjacent machine number:
    c = adj.apply(lambda x: x.index[x != 0.0].to_list(), axis=1).to_list()

    start = timeit.default_timer() # Timer starts
    # search the machine with maximum resource consumption while available 
    # for serving user i now as well
    for i in range (len(users)):
        # compute the minimum radius/resource ratio, and get the machine number for allocation
        max_val = 0
        if len(c[i]) > 0:
            for j in range(len(c[i])):
                if ((w_2  * distance.euclidean(obfu.iloc[i,0:2], bia.iloc[c[i][j],1:3])+1) * users.iloc[i,3:6]* 10000) / (w_3 * servers1.loc[c[i][j],'RADIUS'] * servers1.iloc[c[i][j],11:14].sum()) > max_val: 
                    max_val = ((w_2  * distance.euclidean(obfu.iloc[i,0:2], bia.iloc[c[i][j],1:3])+1) * users.iloc[i,3:6]* 10000) / (w_3 * servers1.loc[c[i][j],'RADIUS'] * servers1.iloc[c[i][j],11:14].sum())
                    num = c[i][j]
        
            # respond user with corresponding neighbour server
            allocation.iloc[i, num] = num
            servers1.iloc[c[i][j],11:14] = servers1.iloc[c[i][j],11:14] - users.iloc[i,3:6]
    
    stop = timeit.default_timer() # Timer ends
    time = stop - start
    # print('Time: ', stop - start)
    return allocation, time
