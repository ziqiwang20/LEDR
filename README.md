# Location Privacy-Preserving Edge Demand Response (LEDR)
Code and dataset for tackling location privacy-preserving edge demand response (LEDR) problem are presented here comprehensively and systematically.

## EDR-Dataset:
Here, we synthesize a new dataset named EDR based on the information from [AWS wavelength](https://aws.amazon.com/wavelength/features/), [Alibaba Cloud](https://github.com/alibaba/clusterdata), and a real-world dataset [EUA](https://github.com/swinedge/eua-dataset), including edge server capacities, edge server coverages, edge server and user locations within the Melbourne CBD area, server start-up and maintenance costs, etc.
 - edge servers: contains datasets of edge server locations and capabilities.
 - users folder: contains datasets of user location and resource demands.
 
## GEES-Code:
The code of GEES is presented systematically. Different stages regarding the LEDR problem are simulated and modeled by respective functions.


### Language Version:
Python 3.8.9

### Packages:
Libraries including Numpy, Pandas, Scipy, POT, and Math are leveraged for computing, while Matplotlib and Timeit are utilized for figure presentation and time recording.

####  Quick Set-up: 
Following are the packages required during the project of GEES. This is a quick set-up to help you get your environment prepared.
```
pip install numpy # numpy 1.25.1
pip install pandas # pandas 1.5.3
pip install scipy # scipy 1.11.1
pip install POT # POT 0.9.0
pip install math 
pip install timeit
```



