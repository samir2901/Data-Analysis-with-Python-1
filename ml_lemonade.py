'''
Machine Learning for the teenager's lemonade sales. After analysing, the data
we can see that the output variable i.e the sales is kind of linearly dependent
on each of the factor which are temperature, number of flyers distributed and the logarithm
of rainfall. We are not considering just rainfall because the sales is exponentially 
decreases with increase of the rainfall. So we better consider log of rainfall. This makes
the relationship between sales and rain linear.

Finally, we make consider estimator function for sales (y_p) as a function of temperature(x1), 
log of rain(x2) and number of flyers distributed.

Therefore, the equation can be written as y_p = b0 + b1*x1 + b2*x2 + b3*x3

'''

import pandas as pd 
import numpy as np
from sklearn import linear_model

df = pd.read_csv('Lemonade.csv')
print('The data frame is')
print(df)
print()
print()
df['Rainfall'] = np.log(df['Rainfall'])  #making log of rainfall
print('Data Frame after taking the log of rainfall')
print()
print(df)

X = df[['Temperature','Rainfall','Flyers']]
y = df['Sales']
reg = linear_model.LinearRegression()
reg.fit(X,y)
b = reg.coef_
b0 = reg.intercept_
rss = reg.score(X,y)
print('The equation can be written as y_p =',b0,'+ (',b[0],'* x1)','+ (',b[1],'* x2)','+ (',b[2],'* x3)')
print('The RSS for this regression is', rss)














