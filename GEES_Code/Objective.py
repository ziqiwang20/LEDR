import numpy as np
import pandas as pd
from scipy.spatial import distance # Euclidean distance
from utils import effective_users

class Objective_Function:
    def __init__(self, users, servers, obfu_matrix, bia_matrix, allo, eff_allo_matrix, w_1, w_2, w_3):
        self.users = users # original users
        self.servers = servers # original servers
        self.obfu_matrix = obfu_matrix # the reuslt of get_obfuscation_location
        self.bia_matrix = bia_matrix # the result of get_BIA_locations
        self.allo = allo # allocation matrix based on bia locations
        self.eff_allo_matrix = eff_allo_matrix # the result of effective_users after SGR
        self.w_1 = w_1; self.w_2 = w_2; self.w_3 = w_3 # adjustment coefficient of three items of the objective

    def F_coverage(self): # System Utility
        return (np.count_nonzero(self.eff_allo_matrix) / len(self.users) * 100)
    
    def F_privacy_abso(self): # Absolute Overall User Privacy in the MEC System
        sum = 0
        a_k = pd.DataFrame(np.zeros((len(self.eff_allo_matrix),1)))
        for i in range(len(self.eff_allo_matrix)):
            # May need to be adjusted!!!
            if np.count_nonzero(self.eff_allo_matrix.iloc[i,:]) != 0:
                self.users.iloc[i,5] = distance.euclidean(self.bia_matrix.iloc[i,0:2], self.users.iloc[i,0:2])
                a_k.iloc[i,0] = (self.users.iloc[i,5] / (self.servers.loc[self.eff_allo_matrix.iloc[i,:].ne(0).idxmax(),'RADIUS'] * 0.00001)) ** 2
            sum = sum + a_k.iloc[i,0]
        return  (1/np.count_nonzero(self.eff_allo_matrix)) * sum * 100

    def F_energy(self): # Energy Efficiency
        users_energy = 0; users_energy_t = 0; servers_energy = 0; servers_energy_t = 0; effective_energy = 0

        # Alter PoD，Running = 1， Pending = Failed = 0
        self.servers = self.servers.replace({'Running': 1, 'Succeeded': 1, 'Pending': 0, 'Failed': 0})
        # Calculate effective energy consumption
        # assume that runcost in the users matrix
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
    
    def calculate_obj(self):
        obj_value = self.w_1 * (self.F_coverage() / 100) + self.w_2 * np.sqrt(self.F_privacy_abso()/100) + self.w_3 * (self.F_energy() / 100)
        return obj_value * 100

