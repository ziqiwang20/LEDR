import numpy as np
import pandas as pd
from scipy.stats import norm, laplace
from scipy.integrate import quad
from math import sqrt, exp
from scipy.spatial import distance

# Section 1: Exponential (Part)
def gaussian(x, loc, scale):
    return np.exp(-((x - loc) ** 2) / (2 * scale ** 2)) / np.sqrt(2 * np.pi * scale)

def exp_Mechanism(lat,lon,loc, scale, eps, sensitivity):
    scores = pd.DataFrame(np.zeros((100,1)))
    for i in range(100):
        scores.iloc[i,0] = gaussian(i, loc, scale)
    
    # Normalization
    scores = scores / np.sum(scores)
    
    # Calculate probability0
    probability0 = np.zeros((100,1))
    region = pd.DataFrame(np.zeros((10 * 10, 3)), columns=['latitude', 'longitude', 'Prob'])

    # 1) generate personalized NPLS
    for i in range(10):
        region.loc[10*i:10*i+10,'latitude'] = lat + 0.0001 * (5 - i) # 10m x 10m
        for j in range(10):
            region.loc[10*i+j,'longitude'] = lon - 0.0001 * (5 + j)
    
    # 2) calculate probability
    for k in range(len(scores)):
        probability0[k,0] = np.exp(0.5 * eps * scores.iloc[k,0] / sensitivity)
    sum_prob = np.sum(probability0)
    region['Prob'] = probability0 / sum_prob
    
    # 3）sampling
    obfu_loc = region.sample(1, weights = region.loc[:,'Prob'].values, replace = True)
    return obfu_loc

# Get obfuscated locations and sort as a dataframe
def get_obfuscation_location(users,loc, scale, eps, sensitivity):
    obfu_matrix = pd.DataFrame(columns=['latitude', 'longitude', 'Prob'])
    for i in range(len(users)):
        lat = users.iloc[i,0]
        lon = users.iloc[i,1]
        obfu_matrix = pd.concat([obfu_matrix,exp_Mechanism(lat,lon,loc, scale, eps,sensitivity)])
    obfu_matrix_full = pd.concat([obfu_matrix.reset_index(),users.iloc[:,2:6]],axis=1).drop(columns = 'index') 
    return obfu_matrix_full

# (2) Laplacian Noise: 
def get_Laplacian_location(users, loc, sensitivity, epsilon):
    users1 = users
    scale = sensitivity / epsilon
    laplace_prob = pd.DataFrame(np.empty((len(users),2)))
    laplace_prob.columns =['laplace_noise', 'Prob']
    
    # define pdf
    def laplace_pdf(x, loc, scale):
        return 1 / (2 * scale) * np.exp(-np.abs(x - loc) / scale)
    Z = np.sum(laplace_pdf(np.linspace(-10, 10, 10000), loc, scale))
    # normalization
    for i in range(len(users)):
        laplace_prob.iloc[i,0] = np.random.laplace(loc, scale)
        laplace_prob.iloc[i,1] = laplace_pdf(laplace_prob.iloc[i,0], loc, scale) / Z
    # concat
    data1 = pd.concat([(users1.iloc[:,0] + laplace_prob.iloc[:,0]), (users1.iloc[:,1] + laplace_prob.iloc[:,0])], axis = 1)
    data1.columns = ['latitude', 'longitude']
    rest = pd.concat([laplace_prob.iloc[:,1],users.iloc[:,2:7]], axis=1)

    laplace = pd.concat([data1, rest], axis=1)
    return laplace


# (3) Gaussian Mechanism:
def get_Gaussian_location(users, mean, epsilon, sensitivity):
    users2 = users
    sigma = sensitivity / epsilon
    gauss_prob = pd.DataFrame(np.empty((len(users),2)))
    gauss_prob.columns =['gauss_noise', 'Prob']
    
    # define pdf
    def gaussian_pdf(x, mean, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    Z = np.sum(gaussian_pdf(np.linspace(-10, 10, 10000), mean, sigma))

    # normalization
    for i in range(len(users)):
        gauss_prob.iloc[i,0] = np.random.normal(mean, sigma)
        gauss_prob.iloc[i,1] = gaussian_pdf(gauss_prob.iloc[i,0], mean, sigma) / Z
        
    # concat
    data1 = pd.concat([(users2.iloc[:,0] + gauss_prob.iloc[:,0]), (users2.iloc[:,1] + gauss_prob.iloc[:,0])], axis = 1)
    data1.columns = ['latitude', 'longitude']
    rest = pd.concat([gauss_prob.iloc[:,1],users.iloc[:,2:7]], axis=1)

    gauss = pd.concat([data1, rest], axis=1)
    return gauss
# Gaussian noise is a loose differential privacy approach, which could guarentee users' privacy to some extent.


# Section 2: Inference Attack
def Bayesian_Inference_Attack(loc_matrix,lat,lon): 
    probability_bia = np.zeros((100,1))
    region_bia = pd.DataFrame(np.zeros((10 * 10, 3)), columns=['latitude', 'longitude', 'Prob'])

    # Defube prior knowledge uniform pi:
    pi_uniform = pd.DataFrame(np.random.uniform(0,1,[100,1])) 
    pi = pi_uniform / np.sum(pi_uniform)
    
    # Gridding:
    for i in range(10):
        region_bia.loc[10*i:10*i+10,'latitude'] = lat + 0.0001 * (5 - i) # 10m x 10m
        for j in range(10):
            region_bia.loc[10*i+j,'longitude'] = lon - 0.0001 * (5 + j)
    
    # Computing probability:
    for k in range(100):
        probability_bia[k,0] = (pi.iloc[k,0] * loc_matrix.iloc[k,2]) / loc_matrix[(loc_matrix['latitude'] == lat) & (loc_matrix['longitude'] == lon)]['Prob'].sum()
    #region_bia = region_bia.assign(Prob = probability_bia)
    region_bia['Prob'] = probability_bia
    
    # Sampling:
    inf_loc = region_bia.sample(1,weights = region_bia.loc[:,'Prob'],replace=True)
    return inf_loc, region_bia

def get_BIA_location(obfu_loc, users):
    inf_matrix_prob = pd.DataFrame(columns=['latitude', 'longitude', 'Prob'])
    for i in range(len(users)):
        lat = obfu_loc.iloc[i,0]
        lon = obfu_loc.iloc[i,1]
        # prob = users.iloc[i,2]
        inf_matrix_prob = pd.concat([inf_matrix_prob, Bayesian_Inference_Attack(obfu_loc,lat,lon)[0]])
    inf_matrix = pd.concat([inf_matrix_prob.reset_index(),users.iloc[:,2:6]],axis=1).drop(columns = 'index')
    return inf_matrix

# Used for calculating the x(o) in the distortion privacy:
def get_individual_BIA_distance(obfu_loc, users, i):
    lat = obfu_loc.iloc[i,0]
    lon = obfu_loc.iloc[i,1]
    region_bia = pd.DataFrame(np.zeros((10*10,2)))
    region_bia = region_bia.rename(columns = {0:'latitude',1:'longitude'})
    min = 1000

    # Gridding:
    for i in range(10):
        region_bia.loc[10*i:10*i+10,'latitude'] = lat + 0.0001 * (5 - i)
        for j in range(10):
            region_bia.loc[10*i+j,'longitude'] = lon - 0.0001 * (5 + j)
    
    # Sorting:
    for j in range(len(region_bia)):
        if distance.euclidean(users.iloc[j,0:2], region_bia.iloc[i,0:2]) < min:
            min = distance.euclidean(users.iloc[i,0:2], region_bia.iloc[i,0:2])
    return min



"""
Demo-test:
(2) Optimal Attack:
# Bayesian based
def Optimal_Attack(users, loc_matrix,lat,lon):
    # users is original locations, loc_matrix is obfuscated locations
    probability_opt = np.zeros((100,1))
    region_opt = pd.DataFrame(np.zeros((10*10,2)))
    region_opt = region_opt.rename(columns = {0:'latitude',1:'longitude'})
    
    # Gridding:
    for i in range(10):
        region_opt.loc[10*i:10*i+10,'latitude'] = lat + 0.00001 * (5 - i)
        for j in range(10):
            region_opt.loc[10*i+j,'longitude'] = lon - 0.00001 * (5 + j)
    
    # Computing probability:
    val = np.empty((100,1))
    for k in range(100):
        # val for pso optimization
        val[k] = pi.iloc[k,0] * loc_matrix.iloc[k,2] * distance.euclidean(users.iloc[k,0:2], region_opt.iloc[k,0:2])
        # pso: obj
        def demo_func(x):
            prob_q = x
            return prob_q * val[k]
        
        probability_opt[k,0] = (pi.iloc[k,0] * loc_matrix.iloc[k,2]) * q * distance.euclidean(users.iloc[k,0:2], region_opt.iloc[k,0:2])
    region_opt = region_opt.assign(Prob = probability_opt)
    
    # Sampling:
    inf_loc = region_opt.sample(1,weights = region_opt.loc[:,'Prob'], replace=True)
    return inf_loc

def get_optimal_location(obfu_loc, users):
    opt_matrix_prob = pd.DataFrame(columns=['latitude', 'longitude', 'Prob'])
    for i in range(len(users)):
        lat = obfu_loc.iloc[i,0]
        lon = obfu_loc.iloc[i,1]
        opt_matrix_prob = opt_matrix_prob.append(Bayesian_Inference_Attack(obfu_loc,lat,lon))
    opt_matrix = pd.concat([opt_matrix_prob.reset_index(),users.iloc[:,2:6]],axis=1).drop(columns = 'index')
    return opt_matrix
"""


