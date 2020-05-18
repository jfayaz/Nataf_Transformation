# -*- coding: utf-8 -*-
"""
author : JAWAD FAYAZ (email: jfayaz@uci.edu)
visit  : https://jfayaz.github.io

------------------------------ Instructions ------------------------------------- 
This code performs Nataf transfomation of two correlated random variables by
approximating their correlation in the standard normal domain (y-space) using 
their marginal distributions and correlation in orginal domain. This code is 
based on the polynomial approximation developed by (Liu and Der Kiureghian, 1986)
described in Appendix B.2 of Melchers (2002)- "Structural Reliability Analysis and Prediction" 


Notations for Marginal Distributions (bother lower and uppercase allowed):
        Normal                                 - 'N'
        Uniform                                - 'U'
        Shifted Exponential                    - 'SE'
        Shifted Rayleigh                       - 'SR'
        Extreme Value I Largest (Gumbel)       - 'G'
        Lognormal                              - 'LN'
        Gamma                                  - 'GM'
        Extreme Value III Smallest (Weibull)   - 'W' 

Note: All combinations of these marginal distributions are not available (Check Melchers 2002)

   
You may run this code in python IDE: 'Spyder' or any other similar IDE
Make sure you have the following python libraries installed:
    pandas 
    numpy
 
INPUT:
The following inputs within the code are required:

        'Dist1'  --> Marginal Distribution of Random Variable 1
        'Dist2'  --> Marginal Distribution of Random Variable 2
        'Corr'   --> Correlation between Random Variable 1 and Random Variable 2 in orginal domain
        'CoV1'   --> Coefficient of Variation (CoV) of Random Variable 1
        'CoV2'   --> Coefficient of Variation (CoV) of Random Variable 2


OUTPUT:
        'CorrY'  --> Correlation between Random Variable 1 and Random Variable 2 in Standard Normal domain (y-space)    
    
%%%%% ========================================================================================================================================================================= %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"""

##### ================== INPUTS  ================== #####

### Marginal Distribution of Random Variable 1
Dist1 = 'gm';

### Marginal Distribution of Random Variable 2
Dist2 = 'LN';

### Correlation between Random Variable 1 and Random Variable 2 in orginal domain
Corr  = 0.9;

### Coefficient of Variation (CoV) of Random Variable 1
CoV1  = 1;

### Coefficient of Variation (CoV) of Random Variable 2
CoV2  = 1;

##### ============ END OF USER INPUTS  ============ #####
#########################################################
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#####

## Importing Libraries
import numpy as np
import pandas as pd

## Function that defines the coefficients of Polynomial based on type of distribution
def Det_Coeff(Dist1,Dist2,Corr,CoV1,CoV2):
    #### If Dist 2 is Normal
    if Dist2.lower() == 'n':
        if Dist1.lower() == 'n':
            a = 1
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'se':
            a = 1.107
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'sr':
            a = 1.014
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0
        elif Dist1.lower() == 'g':
            a = 1.031
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'ln':
            a = CoV2/np.sqrt(np.log(1+CoV2**2))
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0     
        elif Dist1.lower() == 'gm':
            a = 1.001
            b = -.007
            c = 0.118
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0      
        elif Dist1.lower() == 'w':
            a = 1.031
            b = -.195
            c = 0.328
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0  
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')
    
    #### If Dist 2 is Shifted Exponential
    if Dist2.lower() == 'se':
        if Dist1.lower() == 'se':
            a = 1.229
            b = 0
            c = 0
            d = -0.367
            e = 0.153
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'sr':
            a = 1.123
            b = 0
            c = 0
            d = -0.1
            e = 0.021
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0
        elif Dist1.lower() == 'g':
            a = 1.142
            b = 0
            c = 0
            d = -0.154
            e = 0.031
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'ln':
            a = 1.098
            b = 0.019
            c = 0.303
            d = 0.003
            e = 0.025
            f = -0.437
            g = 0
            h = 0
            k = 0
            l = 0     
        elif Dist1.lower() == 'gm':
            a = 1.104
            b = -.008
            c = 0.173
            d = 0.003
            e = 0.014
            f = -0.296
            g = 0
            h = 0
            k = 0
            l = 0      
        elif Dist1.lower() == 'w':
            a = 1.147
            b = 0.145
            c = 0.010
            d = -0.271
            e = 0.459
            f = -0.467
            g = 0
            h = 0
            k = 0
            l = 0  
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')
    
    #### If Dist 2 is Shifted Rayleigh
    if Dist2.lower() == 'sr':
        if Dist1.lower() == 'sr':
            a = 1.028
            b = 0
            c = 0
            d = -0.029
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0
        elif Dist1.lower() == 'g':
            a = 1.046
            b = 0
            c = 0
            d = -0.045
            e = 0.006
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'ln':
            a = 1.011
            b = 0.014
            c = 0.231
            d = 0.001
            e = 0.004
            f = -0.130
            g = 0
            h = 0
            k = 0
            l = 0     
        elif Dist1.lower() == 'gm':
            a = 1.014
            b = -0.007
            c = 0.126
            d = 0.001
            e = 0.002
            f = -0.090
            g = 0
            h = 0
            k = 0
            l = 0      
        elif Dist1.lower() == 'w':
            a = 1.047
            b = -0.212
            c = 0.353
            d = 0.042
            e = 0
            f = -0.136
            g = 0
            h = 0
            k = 0
            l = 0      
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')

    #### If Dist 2 is Type-I Largest (Gumbel)
    if Dist2.lower() == 'g':
        if Dist1.lower() == 'g':
            a = 1.064
            b = 0
            c = 0
            d = -0.069
            e = 0.005
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0 
        elif Dist1.lower() == 'ln':
            a = 1.029
            b = 0.014
            c = 0.233
            d = 0.001
            e = 0.004
            f = -0.197
            g = 0
            h = 0
            k = 0
            l = 0     
        elif Dist1.lower() == 'gm':
            a = 1.031
            b = -0.007
            c = 0.131
            d = 0.001
            e = 0.003
            f = -0.132
            g = 0
            h = 0
            k = 0
            l = 0      
        elif Dist1.lower() == 'w':
            a = 1.064
            b = -0.210
            c = 0.356
            d = 0.065
            e = 0.003
            f = -0.211
            g = 0
            h = 0
            k = 0
            l = 0      
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')
    
    
      
    #### If Dist 2 is Lognormal
    if Dist2.lower() == 'ln':
        if Dist1.lower() == 'ln':
            a = np.log(1+Corr*CoV1*CoV2)/(Corr*np.sqrt(np.log(1+CoV1**2)*np.log(1+CoV2**2)))
            b = 0
            c = 0
            d = 0
            e = 0
            f = 0
            g = 0
            h = 0
            k = 0
            l = 0     
        elif Dist1.lower() == 'gm':
            a = 1.001
            b = 0.004
            c = 0.223
            d = 0.033
            e = 0.002
            f = -0.104
            g = -0.016
            h = 0.130
            k = -0.119
            l = 0.029      
        elif Dist1.lower() == 'w':
            a = 1.031
            b = 0.052
            c = 0.220
            d = 0.052
            e = 0.002
            f = 0.005
            g = -0.210
            h = 0.350
            k = -0.174
            l = 0.009
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')
    
    #### If Dist 2 is Type-I Largest (Gumbel)
    if Dist2.lower() == 'gm': 
        if Dist1.lower() == 'gm':
            a = 1.002
            b = -0.012
            c = 0.125
            d = 0.022
            e = 0.001
            f = -0.077
            g = -0.012
            h = 0.125
            k = -0.077
            l = 0.014      
        elif Dist1.lower() == 'w':
            a = 1.032
            b = -0.007
            c = 0.121
            d = 0.034
            e = 0
            f = -0.006
            g = -0.202
            h = 0.339
            k = -0.111
            l = 0.003      
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')

    #### If Dist 2 is Type-III Smallest (Weibull)
    if Dist2.lower() == 'w':       
        if Dist1.lower() == 'w':
            a = 1.063
            b = -0.200
            c = 0.337
            d = -0.004
            e = -0.001
            f = 0.007
            g = -0.200
            h = 0.337
            k = 0.007
            l = -0.007  
        else:
            raise ValueError('Transformation not found! Distribution : "'+Dist1+'" (Dist1) cannot be used with Distribution: "'+Dist2+'" (Dist2). Check Melchers (2002) for more details.')
    
    return a,b,c,d,e,f,g,h,k,l

### Computing the Coefficients of Polynomial 
[a,b,c,d,e,f,g,h,k,l] = Det_Coeff(Dist1,Dist2,Corr,CoV1,CoV2)

### Ratio of Correlations
R       = a + b*CoV1 + c*CoV1**2 + d*Corr + e*Corr**2 + f*Corr*CoV1 + g*CoV2 + h*CoV2**2 + k*Corr*CoV2 + l*CoV1*CoV2

### Correlations in Standard Normal Space (y-space)
CorrY  = R*Corr
