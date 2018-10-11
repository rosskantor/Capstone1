Background
Automobile finance is highly competitive.  Savvy consumers can price loans online easily filtering competitive rates into acceptable or unacceptable categories.  Lenders rank bought online leads (in addition to those generated in house) in order to contact the best potential clients. Accurate algorithms are critical to contacting the right customer.

Data for this project originated from a mid-size automobile finance company.  The company acts as a middle man, a broker, steering leads to financial institutions.

The Set Up Problem

Data provided for this project is not identified by name or purpose.  All column headings are titled Variable1, Variable2, ..., Result01. An educated guess easily distinguishes a few fields.  Variable 6 is clearly a state column, Variable 3 is categorical, Variable 5 is credit score, Variable11 is probably loan amount and Variable07 may be primary borrower age.


![Missing Data Image](Images/MissingData.png)


Data is sparse for variables 2, 7-11.  Variable 5 is missing a few rows.

 ![Nulls](\Images\NULLS.png)

High Level Statistics for Cleaned Data


 ![High Level Statistics](\Images\HLS.png)


K nearest neighbor(5) was chosen to fill in blank numeric fields.  Because of RAM limitations the data was split into three separate dataframes comprising approximately 10,000 before running KNN.

States were broken divided into region six regions each of which was assigned to a dummy classification.

![Loan Size to Purchase](\Images\Loan Size to Purchase.png)

![Correlation](\Images\Corr2.png)
![Correlation](\Images\Corr1.png)



Below is the variance inflation factor for the y intercept and the 14 feature variables.  Two variables were cut after trial one due to VIF's in excess of 5.

![VIF](\Images\VIF.png)

The Problem

Can I predict loan originations given input data? A logistic model was developed to predict true/false.  



coefs first pass array([ 0.09536577, -0.11794106, -0.04090294, -0.00426806, -0.0267169 ,
        0.07652899,  0.02245709,  0.05656126,  0.00590679, -0.00576597,
        0.00151953,  0.00341846,  0.00266661])

![Cost Benefit](\Images\Cost Benefit.png)

When all variables (except Region_MT) are included the profit matrix and ROC curve are displayed below.
![Profit Model](\Images\All Variables.png)

![Confusion Matrix (All inputs)](\Images\Confusion Matrix 1.png)

Conclusion

Supplied inputs are not highly correlated with with borrower preferences.  Borrower behavior for this firm appears to be random.  Credit score, loan amount, age, etc are not good predictors of borrower preference.

Lessons Learned

* Be a better programmer
* Experience with data sets and regression modeling prior would help.
* Predicting negatives is important too
