import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from scipy.spatial import distance  # Euclidean distance
from EESaver import EESaver_test
from SGR import Secure_Greedy_Response
from utils import adjacent_servers, dominant_resource_demand, effective_users

class Objective_np:
    def __init__(self, users, servers, allo, eff_allo_matrix, w_1, w_2, w_3):
        self.users = users # original users
        self.servers = servers # original servers
        self.allo = allo
        self.eff_allo_matrix = eff_allo_matrix
        self.w_1 = w_1; self.w_2 = w_2; self.w_3 = w_3 # weighted coefficient

    def F_coverage(self): # Coverage item
        return (np.count_nonzero(self.eff_allo_matrix) / len(self.users) * 100)

    def F_energy(self): # Energy item
        users_energy = 0; users_energy_t = 0; servers_energy = 0; servers_energy_t = 0; effective_energy = 0
        self.servers = self.servers.replace({'Running': 1, 'Succeeded': 1, 'Pending': 0, 'Failed': 0})
        
        # Calculate effective energy consumption
        for k in range(len(self.eff_allo_matrix)):
            for j in range(len(self.servers)):
                if np.count_nonzero(self.eff_allo_matrix.iloc[k,:]) != 0 :
                    servers_energy = servers_energy - self.servers.loc[j,'Alpha'] * (self.servers.loc[j,'PoD'] - 1) + self.servers.loc[j,'Tau'] * self.servers.loc[j,'PoD']
                j = j + 1
            users_energy = users_energy + np.count_nonzero(self.eff_allo_matrix.iloc[k,:]) * self.users.loc[k,'Run_Cost']
            k = k + 1
        effective_energy = servers_energy + users_energy
    
        # Calculate total energy consumption
        for k in range(len(self.users)):
            for j in range(len(self.servers)):
                servers_energy_t = servers_energy_t - self.servers.loc[j,'Alpha'] * (self.servers.loc[j,'PoD'] - 1) + self.servers.loc[j,'Tau'] * self.servers.loc[j,'PoD']
                j = j + 1
            users_energy_t = users_energy_t + np.count_nonzero(self.allo.iloc[k,:]) * self.users.loc[k,'Run_Cost']
            k = k + 1
        total_energy = servers_energy_t + users_energy_t
        return (effective_energy / total_energy * 100)


# Initialization:
Servers = pd.read_csv("/Your-Own-Path/servers.csv")
Users = pd.read_csv("/Your-Own-Path/users.csv")

# -----Main Experiments:-----
# (1) GEES:
# Timer:
# start = timeit.default_timer()
def GEES_np(users, servers, w_1, w_2, w_3):
    sgr,time = Secure_Greedy_Response(users, servers, w_2, w_3)
    allo = effective_users(sgr, users, servers)
    obj = Objective_np(users, servers, sgr, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    return obj.F_coverage(), obj.F_energy(), time
# stop = timeit.default_timer()
# print('Time: ', stop - start)

# (2) EESaver:
# start = timeit.default_timer()
def EESaver_np(users, servers, w_1, w_2, w_3):
    eesaver, time = EESaver_test(users, servers)
    allo = effective_users(eesaver, users, servers)
    obj = Objective_np(users, servers, eesaver, allo, w_1, w_2, w_3)
                       
    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    return obj.F_coverage(), obj.F_energy(), time
# stop = timeit.default_timer()
# print('Time: ', stop - start)