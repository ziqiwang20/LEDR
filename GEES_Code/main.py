import sys
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial import distance  # Euclidean distance
from EESaver import EESaver_test
from Objective import Objective_Function
from Privacy_Game_ori import (get_BIA_location, get_Gaussian_location,
                          get_Laplacian_location, get_obfuscation_location)
from SGR import Secure_Greedy_Response
from utils import adjacent_servers, dominant_resource_demand, effective_users
sys.path.append('/Your-Own-Path')

# Initialization: dataset
Servers = pd.read_csv("/Your-Own-Path/servers.csv")
Users = pd.read_csv("/Your-Own-Path/users.csv")

# -----Main Experiments:-----
# (1) GEES:
# Timer:
# start = timeit.default_timer()
def GEES(users, servers, loc, scale, eps, sensitivity, w_1, w_2, w_3):
    obfu = get_obfuscation_location(users, loc, scale, eps, sensitivity)
    bia = get_BIA_location(obfu, users)
    sgr,time = Secure_Greedy_Response(bia, servers,obfu,bia, w_2, w_3)
    allo = effective_users(sgr, users, servers)
    obj = Objective_Function(users, servers, obfu, bia, sgr, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Effieciency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(), obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time
# stop = timeit.default_timer()
# print('Time: ', stop - start)


# (2) Laplace + Secure Greedy Response:
def Laplace_SDR(users, servers, eps, sensitivity, w_1, w_2, w_3):
    obfu_lap = get_Laplacian_location(users, 0, sensitivity, eps)
    bia = get_BIA_location(obfu_lap, users)
    sgr_laplace, time = Secure_Greedy_Response(bia, servers, obfu_lap,bia, w_2, w_3)
    allo = effective_users(sgr_laplace, users, servers)
    obj = Objective_Function(users, servers, obfu_lap, bia, sgr_laplace, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(),obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time


# (3) Gaussian + Secure Greedy Response:
def Gaussian_SDR(users, servers, loc, eps, sensitivity, w_1, w_2, w_3):
    obfu_gauss = get_Gaussian_location(users, loc, eps, sensitivity)
    bia = get_BIA_location(obfu_gauss, users)
    sgr_gaussian,time = Secure_Greedy_Response(bia, servers,obfu_gauss,bia, w_2, w_3)
    allo = effective_users(sgr_gaussian, users, servers)
    obj = Objective_Function(users, servers, obfu_gauss, bia, sgr_gaussian, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(), obj.F_privacy_rela(), obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time


# (4) PSO + EESaver:
def PSO_EESaver(users, servers, loc, scale, eps, sensitivity, w_1, w_2, w_3):
    obfu_ee = get_obfuscation_location(users, loc, scale, eps, sensitivity)
    bia = get_BIA_location(obfu_ee, users)
    eesaver, time = EESaver_test(bia, servers)
    allo = effective_users(eesaver, users, servers)
    obj = Objective_Function(users, servers, obfu_ee, bia, eesaver, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(), obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time

# (5) Laplace + EESaver:
def Laplacian_EESaver(users, servers, eps, sensitivity, w_1, w_2, w_3):
    obfu_lapee = get_Laplacian_location(users, 0, sensitivity, eps)
    bia = get_BIA_location(obfu_lapee, users) 
    lap_eesaver, time = EESaver_test(obfu_lapee, servers)
    allo = effective_users(lap_eesaver, users, servers)
    obj = Objective_Function(users, servers, obfu_lapee, bia, lap_eesaver, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(), obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time

# (6) Gaussian + EESaver:
def Gaussian_EESaver(users, servers, loc, eps, sensitivity, w_1, w_2, w_3):
    obfu_gauss = get_Gaussian_location(users, loc, eps, sensitivity)
    bia = get_BIA_location(obfu_gauss, users)
    eesaver, time = EESaver_test(bia, servers)
    allo = effective_users(eesaver, users, servers)
    obj = Objective_Function(users, servers, obfu_gauss, bia, eesaver, allo, w_1, w_2, w_3)

    print("Coverage Score is :", obj.F_coverage())   # System Utility
    print("Absolute Privacy Score is :", obj.F_privacy_abso()) # Absolute Privacy
    print("Energy Score is :", obj.F_energy()) # Energy Efficiency
    print("Overall Score is :", obj.calculate_obj()) # Overall Objective Score
    return obj.F_coverage(), obj.F_privacy_abso(), obj.F_energy(), obj.calculate_obj(), time