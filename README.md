# Nataf_Transformation

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
