Background
Data for this project originated from a local automobile finance company.  The company matches borrowers to credit unions.  


The Set Up Problem

No data field has been identified by its name or purpose.  An educated guess must be made as to meaning of each field.  A few fields are probably easy to deduce:  Variable 6 is clearly a state column, Variable11 is probably loan amount and Variable07 may be primary borrower age.


A number of data fields contained NaN.

Variable09  Variable10  Variable11  Result01  Result02  Result03 Region  
0         -0.0087       -3.53    21547.00       1.0         0         0     WE  
1             NaN         NaN    18050.40       2.5         1         1     MT  
2          0.0241      276.68         NaN       2.5         0         0     NE  
3          0.1204      268.81         NaN       1.0         0         0      S  
4             NaN         NaN         NaN       1.0         0         0     NE  
5          0.0822       57.38         NaN       1.0         0         0      P  
6             NaN         NaN         NaN       1.0         0         0      P  
7             NaN         NaN         NaN       1.0         0         0     MW  
8             NaN         NaN         NaN       1.0         0         0      S  
9             NaN         NaN         NaN       1.0         0         0      S  
10         0.1245      164.09     3811.00       1.0         0         0      P  
11            NaN         NaN         NaN       1.0         0         0      S  
12        -0.0082      134.69         NaN       2.5         0         0     NE   


K nearest neighbor(5) was chosen to fill in blank numeric fields.  Because of RAM limitations the data was split into three separate dataframes comprising approximately 10,000 before running KNN.

States were broken divided into region seven regions each of which was assigned to a dummy classification.

Pearson R
Variable01 (0.1646408496812585, 2.89606744386408e-197)
Variable02 (-0.08174999276992656, 1.4367617059312514e-49)
Variable03 (-0.19925944561704992, 8.17963230603482e-290)
Variable04 (0.2378841809993534, 0.0)
Variable05 (-0.07383037190000141, 9.911151575186567e-41)
Variable07 (-0.009853147147902586, 0.0749083457304896)
Variable08 (-0.03539776388258389, 1.5512961851600497e-10)
Variable09 (0.11054451922438226, 2.339300875667538e-89)
Variable10 (0.03774820536317447, 8.760831161453883e-12)
Variable11 (0.07626713295439226, 2.3582734451910936e-43)
Result01 (0.3728748414590001, 0.0)
Result02 (1.0, 0.0)
Result03 (0.3359300048706456, 0.0)


Coefficient Matrix:

([[-1.71987234e-04, -1.90388557e-02, -5.56688045e-04,
        -1.15124270e-02, -5.98296464e-07,  3.22275177e-05,
        -3.48823640e-04,  1.67799614e-05]])


Below is the variance inflation factor for the y intercept and the 14 feature variables.  Two variables were cut after trial one due to VIF's in excess of 5.


VIF
VIF Factor    features
0    83.046455   Intercept
1     2.212373  Variable01
2     2.215403  Variable03
3     1.063850  Variable05
4     1.224052  Variable07
5     1.072015  Variable08
6     1.057516  Variable09
7     1.221206  Variable10
8     1.162699  Variable11
9     2.458582   Region_MW
10    2.553546   Region_NE
11    2.281210    Region_P
12    1.050682   Region_PL
13    3.399661    Region_S
14    1.084799   Region_WE


Results: Logit
=================================================================
Model:              Logit            No. Iterations:   7.0000    
Dependent Variable: Result02         Pseudo R-squared: 0.069     
Date:               2018-10-08 15:58 AIC:              32535.2548
No. Observations:   32674            BIC:              32661.1698
Df Model:           14               Log-Likelihood:   -16253.   
Df Residuals:       32659            LL-Null:          -17459.   
Converged:          1.0000           Scale:            1.0000    
------------------------------------------------------------------
Coef.   Std.Err.     z      P>|z|    [0.025   0.975]
------------------------------------------------------------------
Intercept    -2.0918    0.1292  -16.1887  0.0000  -2.3451  -1.8386
Variable01    0.4072    0.0511    7.9734  0.0000   0.3071   0.5073
Variable03   -0.0135    0.0006  -21.5842  0.0000  -0.0147  -0.0123
Variable05   -0.0002    0.0000   -6.7731  0.0000  -0.0002  -0.0001
Variable07   -0.0061    0.0012   -5.1739  0.0000  -0.0084  -0.0038
Variable08   -0.0000    0.0000   -4.7643  0.0000  -0.0000  -0.0000
Variable09    5.7052    0.2959   19.2782  0.0000   5.1251   6.2852
Variable10    0.0001    0.0002    0.3091  0.7572  -0.0003   0.0004
Variable11    0.0000    0.0000   18.5605  0.0000   0.0000   0.0000
Region_MW     0.2528    0.0579    4.3626  0.0000   0.1392   0.3663
Region_NE     0.1503    0.0574    2.6211  0.0088   0.0379   0.2627
Region_P      0.2252    0.0599    3.7570  0.0002   0.1077   0.3427
Region_PL     0.5711    0.1891    3.0205  0.0025   0.2005   0.9417
Region_S      0.2291    0.0524    4.3739  0.0000   0.1265   0.3318
Region_WE     0.2052    0.1542    1.3304  0.1834  -0.0971   0.5074
=================================================================
