---
layout: post
title: Python for...Wine Analysis?
blurb: What makes a wine special? What makes a wine a truly **great** wine? Using some simple Python, today we're going to find out.
tags:
    - machine learning
    - python
    - pandas
    - scikit-learn
---

## Imports

```python

import pandas as pd
from pandas import DataFrame, Series
from patsy import dmatrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import tree
import statsmodels.api as sm
import os

%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib

# OLS Regression

Determining wine quality from all other factors

## Setup

Assemble the red and white dataframes, then concatenate them into a master dataframe.


```python
# Load red wines
names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'color']

dfRed = pd.read_csv('data/winequality-red.csv', sep=";")
dfRed['color'] = 'red'
dfRed.columns = names

# Load white wines
dfWhite = pd.read_csv('data/winequality-white.csv', sep=";")
dfWhite['color'] = 'white'
dfWhite.columns = names

# Create master dataframe
df = pd.concat([dfRed, dfWhite])
df = df.reset_index(drop=True)

df[:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>



Create formula for the quality of a wine from all other variables


```python
# Exclude color (categorical) and quality from regressing on quality
traitsToExclude = ['color', 'quality']

# Generate formula
initialFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in traitsToExclude])
initialFormula
```




    'quality ~ 0 + fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'



## Red Wines

Run regression on red wines


```python
def ols(data, formula):
    # Load inital design matrix
    y, X = dmatrices(formula, data=data, return_type='dataframe')

    # Fit initial model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = sm.OLS(y_train, X_train)
    result = model.fit()
    return result

result = ols(dfRed, initialFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.987
    Model:                            OLS   Adj. R-squared:                  0.987
    Method:                 Least Squares   F-statistic:                     7731.
    Date:                Mon, 01 May 2017   Prob (F-statistic):               0.00
    Time:                        17:02:22   Log-Likelihood:                -1102.4
    No. Observations:                1119   AIC:                             2227.
    Df Residuals:                    1108   BIC:                             2282.
    Df Model:                          11                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    fixed_acidity            0.0107      0.019      0.556      0.579      -0.027       0.048
    volatile_acidity        -1.2014      0.145     -8.304      0.000      -1.485      -0.918
    citric_acid             -0.2845      0.177     -1.609      0.108      -0.632       0.062
    residual_sugar           0.0006      0.015      0.042      0.967      -0.029       0.031
    chlorides               -1.8694      0.503     -3.718      0.000      -2.856      -0.883
    free_sulfur_dioxide      0.0064      0.003      2.409      0.016       0.001       0.012
    total_sulfur_dioxide    -0.0039      0.001     -4.380      0.000      -0.006      -0.002
    density                  4.6683      0.743      6.285      0.000       3.211       6.126
    pH                      -0.5557      0.191     -2.916      0.004      -0.930      -0.182
    sulphates                0.8421      0.132      6.361      0.000       0.582       1.102
    alcohol                  0.3019      0.021     14.598      0.000       0.261       0.342
    ==============================================================================
    Omnibus:                       18.922   Durbin-Watson:                   2.017
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.011
    Skew:                          -0.159   Prob(JB):                     8.27e-07
    Kurtosis:                       3.707   Cond. No.                     2.40e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.4e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
# Looks great! But we can do better
# Create new formula and fit new design matrix
exemptions = ['color', 'quality', 'fixed_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'pH', 'chlorides']
newFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in exemptions])
result = ols(dfRed, newFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.987
    Model:                            OLS   Adj. R-squared:                  0.987
    Method:                 Least Squares   F-statistic:                 2.045e+04
    Date:                Mon, 01 May 2017   Prob (F-statistic):               0.00
    Time:                        17:02:22   Log-Likelihood:                -1127.5
    No. Observations:                1119   AIC:                             2263.
    Df Residuals:                    1115   BIC:                             2283.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ====================================================================================
                           coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------------
    volatile_acidity    -1.2977      0.118    -11.030      0.000      -1.529      -1.067
    density              2.6654      0.236     11.303      0.000       2.203       3.128
    sulphates            0.6316      0.120      5.256      0.000       0.396       0.867
    alcohol              0.3131      0.019     16.721      0.000       0.276       0.350
    ==============================================================================
    Omnibus:                       17.902   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.023
    Skew:                          -0.123   Prob(JB):                     8.22e-07
    Kurtosis:                       3.735   Cond. No.                         134.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# Looks great!
result.params
```




    volatile_acidity   -1.297727
    density             2.665419
    sulphates           0.631601
    alcohol             0.313136
    dtype: float64



## White Wines


```python
# Recreate master formula
traitsToExclude = ['color', 'quality']
initialFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in traitsToExclude])

result = ols(dfWhite, initialFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.984
    Model:                            OLS   Adj. R-squared:                  0.983
    Method:                 Least Squares   F-statistic:                 1.853e+04
    Date:                Mon, 01 May 2017   Prob (F-statistic):               0.00
    Time:                        17:02:22   Log-Likelihood:                -3933.1
    No. Observations:                3428   AIC:                             7888.
    Df Residuals:                    3417   BIC:                             7956.
    Df Model:                          11                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    fixed_acidity           -0.0547      0.018     -2.986      0.003      -0.091      -0.019
    volatile_acidity        -2.0939      0.135    -15.483      0.000      -2.359      -1.829
    citric_acid             -0.0507      0.115     -0.441      0.659      -0.276       0.175
    residual_sugar           0.0276      0.003      8.880      0.000       0.022       0.034
    chlorides               -1.2006      0.680     -1.766      0.078      -2.534       0.133
    free_sulfur_dioxide      0.0038      0.001      3.809      0.000       0.002       0.006
    total_sulfur_dioxide    -0.0005      0.000     -1.004      0.316      -0.001       0.000
    density                  2.0509      0.424      4.836      0.000       1.219       2.883
    pH                       0.1571      0.101      1.559      0.119      -0.041       0.355
    sulphates                0.3709      0.118      3.155      0.002       0.140       0.601
    alcohol                  0.3744      0.014     27.576      0.000       0.348       0.401
    ==============================================================================
    Omnibus:                       87.955   Durbin-Watson:                   2.036
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              203.502
    Skew:                           0.064   Prob(JB):                     6.46e-45
    Kurtosis:                       4.187   Cond. No.                     7.99e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 7.99e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
# Optimize result
traitsToExclude = ['color', 'quality', 'volatile_acidity', 'citric_acid', 'chlorides', 'density', 'pH' ,'sulphates', 'total_sulfur_dioxide']
newFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in traitsToExclude])
result = ols(dfWhite, newFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.981
    Model:                            OLS   Adj. R-squared:                  0.981
    Method:                 Least Squares   F-statistic:                 4.536e+04
    Date:                Mon, 01 May 2017   Prob (F-statistic):               0.00
    Time:                        17:02:22   Log-Likelihood:                -4132.6
    No. Observations:                3428   AIC:                             8273.
    Df Residuals:                    3424   BIC:                             8298.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    fixed_acidity           0.0548      0.013      4.307      0.000       0.030       0.080
    residual_sugar          0.0308      0.003     10.354      0.000       0.025       0.037
    free_sulfur_dioxide     0.0074      0.001      9.177      0.000       0.006       0.009
    alcohol                 0.4788      0.008     61.853      0.000       0.464       0.494
    ==============================================================================
    Omnibus:                       99.768   Durbin-Watson:                   2.027
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              250.507
    Skew:                          -0.049   Prob(JB):                     4.01e-55
    Kurtosis:                       4.321   Cond. No.                         44.3
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
result.params
```




    fixed_acidity          0.054813
    residual_sugar         0.030799
    free_sulfur_dioxide    0.007426
    alcohol                0.478847
    dtype: float64



# Logistic Regression


```python
# Determine color from other traits

# Create formula to identify color
formula = "color ~ 0 + " + " + ".join([column for column in df if column != 'color'])
formula
```




    'color ~ 0 + fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol + quality'




```python
# Split up design matrices
Y, X = dmatrices(formula, data=df, return_type="dataframe")
# Since color is a binary categorical variable, we can look just for red
y = Y['color[red]'].values

# Split into test data and fit model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LogisticRegression()
result = model.fit(X_train, y_train)
```


```python
# Score model - nice!
prediction = model.predict(X_test)
metrics.accuracy_score(y_test, prediction)
```




    0.97692307692307689




```python
# What's important?
weights = Series(model.coef_[0], index=X.columns.values)
weights.sort_values()
```




    density                -2.595855
    citric_acid            -0.919365
    alcohol                -0.740090
    quality                -0.189836
    residual_sugar         -0.145464
    total_sulfur_dioxide   -0.064688
    free_sulfur_dioxide     0.045683
    fixed_acidity           0.705674
    pH                      2.083909
    chlorides               2.260608
    sulphates               6.613474
    volatile_acidity        8.025657
    dtype: float64




```python
# Can we make a chart to figure out how great a wine is?
traitsToExclude = ['color', 'quality']
formula = 'C(quality) ~ 0 + C(color) + ' + " + ".join([column for column in df if column not in traitsToExclude])
formula
```




    'C(quality) ~ 0 + C(color) + fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol'




```python
for idf in [dfRed, dfWhite]:
    nums = {}
    ratings = idf.quality.unique()
    for num in ratings:
        Y, X = dmatrices(formula, idf, return_type='dataframe')
        y = Y["C(quality)[%d]" % num].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        model = tree.DecisionTreeClassifier(criterion='entropy')
        result = model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        nums[num] = metrics.accuracy_score(y_test, prediction)
    print nums
    numdf = DataFrame(nums.items(), columns=["Rating", "Accuracy"])
    numdf.plot(x="Rating", y="Accuracy").set_xlabel("%s Ratings" % idf.color.unique()[0].title())
```

    {3: 0.9916666666666667, 4: 0.93333333333333335, 5: 0.70208333333333328, 6: 0.68125000000000002, 7: 0.87916666666666665, 8: 0.98333333333333328}
    {3: 0.9938775510204082, 4: 0.95238095238095233, 5: 0.78095238095238095, 6: 0.67278911564625854, 7: 0.8136054421768707, 8: 0.95170068027210886, 9: 0.99863945578231295}



![png](Final%20Project_files/Final%20Project_21_1.png)



![png](Final%20Project_files/Final%20Project_21_2.png)



```python
tree.export_graphviz(model, feature_names=X.columns)
os.system('dot -Tpng tree.dot -o tree.png')
```

    /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sklearn/tree/export.py:386: DeprecationWarning: out_file can be set to None starting from 0.18. This will be the default in 0.20.
      DeprecationWarning)





    0



![9](tree.png)


```python

```


```python
print "Red Qualities"
print dfRed['quality'].value_counts()
print ""
print "White Qualities"
print dfWhite['quality'].value_counts()
```

    Red Qualities
    5    681
    6    638
    7    199
    4     53
    8     18
    3     10
    Name: quality, dtype: int64
    
    White Qualities
    6    2198
    5    1457
    7     880
    8     175
    4     163
    3      20
    9       5
    Name: quality, dtype: int64



```python

```