import numpy as np
import pandas as pd
from scipy.spatial import distance # Euclidean distance

def adjacent_servers(users, servers): # N users, M servers
    i = 0; j = 0
    adj_ser = pd.DataFrame(np.zeros((len(users),len(servers)))) 
    users_location = users.iloc[:,0:2]
    servers_location = servers.iloc[:,1:3]
    for i in range(len(users)):
        for j in range(len(servers)):
            if distance.euclidean(users_location.iloc[i,:], servers_location.iloc[j,:]) <= servers.loc[j,"RADIUS"] * 0.00001 :
                adj_ser.loc[i,j] = j
            else:
                adj_ser.loc[i,j] = 0
            j = j + 1
        i = i + 1
    return adj_ser


def effective_users(allo_mat, users, servers): # Based on the definition
    i = 0
    users_location = users.iloc[:,0:2]
    servers_location = servers.iloc[:,1:3]
    allo_1 = allo_mat
    for i in range(len(allo_1)):
        num = allo_1.iloc[i,:].ne(0).idxmax()
        if distance.euclidean(users_location.iloc[i,:], servers_location.iloc[num,:]) >= servers.loc[num, 'RADIUS'] * 0.00001:
            allo_1.iloc[i,num] = 0
    return allo_1

def dominant_resource_demand(users, servers, adj):
    servers = servers.assign(RESO = servers.iloc[:,11:13].sum(axis=1))
    users = users.assign(INDEX = np.arange(0,len(users)),
                         RESO = users.iloc[:,2:4].sum(axis = 1),
                         aDRD = np.zeros((len(users),1)))
    
    i = 0; j = 0; t = 0; r = 0
    for i in range(len(users)):
        for j in range(len(servers)):
            if adj.iloc[i,j] != 0 and servers.loc[adj.iloc[i,j], 'RESO'] > r:
                users['aDRD'][i] = servers.loc[adj.iloc[i,j], 'RESO']
                r = users['aDRD'][i]
            j = j + 1
        i = i + 1

    users = users.assign(DRD = users['RESO'] / servers.loc[adj.iloc[i,j], 'RESO'])
    users = users.sort_values('DRD', ascending = False)
    users = users.reset_index().drop(columns = 'index')
    return users
