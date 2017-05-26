---
layout: post
published: false
title: Python for...wine tasting?
blurb: What makes a wine special? What makes a wine a truly **great** wine? Using some simple Python, today we're going to find out.
tags:
    - machine learning
    - python
    - pandas
    - scikit-learn
---

## Introduction

Originally written for my Advanced Analytics Programming class at The University of Texas at Austin, this article is meant to quickly show some neat things Machine Learning can provide on the wonderful world of wine. This article assumes a basic knowledge of Machine Leaning, but feel free to browse through the insights.

I'll be doing an analysis on the [Wine Quality Data Set](http://archive.ics.uci.edu/ml/datasets/Wine+Quality) provided by the UCI Machine Learning Repository. I'm not a wine expert by any means, but I wanted to see what really goes into a quality wine for my next purchase. The dataset contains data about 1,599 red wines and 4,898 white wines. Each wine has these features:  

* **Fixed acidity** - Acidity contained in the grapes
* **Volatile acidity** - Acidity caused by fermentation of the wine
* **Citric acid** - Catalyst for fermentation
* **Residual sugar** - yum
* **Chlorides** - Chlorine compound who's content typically determined by wines terroir (terr-wah - or where the wine was * grown, important in determining wine origins)
* **Free sulfur dioxide** - Buffer against microbes and oxidation
* **Total sulfur dioxide** - used as a preservative
* **Density** - Weight per liter
* **pH** - base or acid
* **Sulphates** - Preservatives
* **Alcohol** - the fun part
* **Quality** - median of 3 evaluations made by wine experts

Let's get started!

## Setup

### Imports

Each of these imports is a highly valuable resource for machine learning. I recommend looking into these: 

* [pandas - Data Analysis Library](http://pandas.pydata.org/)
* [patsy - Describing statistical models in Python](https://patsy.readthedocs.io/en/latest/)
* [StatsModels: Statistics in Python](http://www.statsmodels.org/stable/index.html)
* [scikit-learn: machine learning in Python](http://scikit-learn.org/stable/)

I've also touched on what I use each import for in my comments.


```python
# Imports

# Data analysis tools
import pandas as pd # Powerful data analysis library
from pandas import DataFrame, Series # Easy access to pandas datastructures
from patsy import dmatrices # Simple way to transform data for analysis
from sklearn.model_selection import train_test_split # Split data for training and testing

# Machine learning tools
import statsmodels.api as sm # Where OLS lives
from sklearn.linear_model import LogisticRegression # For running a logistic regress
from sklearn import tree # For creating a Decision Tree
from sklearn import metrics # For finding out how accurate everything is

# Other
import os

%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib


### Data Assembly

I'm a big fan of documentation, so I guess I'll keep commenting (nearly) every line. Here we begin to assemble our dataset. We don't have much cleanup to do, but the data was provided with a funky seperator and we need to label each whine as red or white. We can then combine our datasets into a unified dataset (mainly for later).


```python
# Change all names to include underscores for patsy formulas
names = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality', 'color']

# Load red wines
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

# Get some basic information

print "Red wines: ", len(dfRed)
print "White wines: ", len(dfWhite)
print "Total wines: ", len(df)
```

    Red wines:  1599
    White wines:  4898
    Total wines:  6497


## OLS Regression

First, we're going to be running an [Ordinary Least Squares Regression](https://en.wikipedia.org/wiki/Ordinary_least_squares#Classical_linear_regression_model) on our data, attempting to find the quality of a wine from all other factors. This will allow us to see which variables are important to the quality of the wine. We can also see which variables are the most significant, and remove those that aren't.

Patsy, which is what we'll use for our design matrices, accepts a '[formula-like](http://patsy.readthedocs.io/en/latest/API-reference.html)' argument so it can separate our data for us. To create our formula, we're going to regress quality on every variable except color (let's look at purely quantifiable variables for now) and quality (for obvious reasons). Since dfRed and dfWhite contain the same feature names, and df is just dfRed and dfWhite combined, we can use the columns in df to create our formula.


```python
# Exclude color (categorical) and quality from regressing on quality
traitsToExclude = ['color', 'quality']

# Generate formula
initialFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in traitsToExclude])
print initialFormula
```

    quality ~ 0 + fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol


### Red Wines

Now that we have our formula, it's time to create a regression. I've defined a simple function that does it all in one step - it will regress a given formula on a given DataFrame. Inside the function, there's a few moving pieces. Let's split the data into an X and y dataframe so that it can be regressed. We then split that data into training and testing data, fit the model, and return our results.


```python
def ols(data, formula):
    """
    ols(data, formula) runs an ordinary least squares regression on a set of data given a formula
    data - pandas.DataFrame
    formula - patsy formula-like
    """
    
    # Load design matrix
    y, X = dmatrices(formula, data=data, return_type='dataframe')

    # Fit model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    result = sm.OLS(y_train, X_train).fit()
    
    return result

result = ols(dfRed, initialFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.987
    Model:                            OLS   Adj. R-squared:                  0.987
    Method:                 Least Squares   F-statistic:                     7731.
    Date:                Thu, 11 May 2017   Prob (F-statistic):               0.00
    Time:                        00:01:52   Log-Likelihood:                -1102.4
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


This looks pretty good - but let's take a look at those p-values. It's clear that many of these variables aren't very significant towards quality, and so we can remove them from our regression.

| **Item** | **P-Value** |
| ----- | ----- |
| Fixed Acidity | 0.579 |
| Citric Acid | 0.108 |
| Residual Sugar | 0.967 |
| Free Sulfur Dioxide | 0.016 |


```python
exemptions = ['color', 'quality', 'fixed_acidity', 'citric_acid', 'residual_sugar', 'free_sulfur_dioxide']
newFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in exemptions])
result = ols(dfRed, newFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.987
    Model:                            OLS   Adj. R-squared:                  0.987
    Method:                 Least Squares   F-statistic:                 1.208e+04
    Date:                Thu, 11 May 2017   Prob (F-statistic):               0.00
    Time:                        00:28:35   Log-Likelihood:                -1107.5
    No. Observations:                1119   AIC:                             2229.
    Df Residuals:                    1112   BIC:                             2264.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ========================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------
    volatile_acidity        -1.1082      0.122     -9.116      0.000      -1.347      -0.870
    chlorides               -2.0664      0.477     -4.330      0.000      -3.003      -1.130
    total_sulfur_dioxide    -0.0027      0.001     -4.308      0.000      -0.004      -0.001
    density                  4.3390      0.469      9.250      0.000       3.419       5.259
    pH                      -0.4287      0.136     -3.147      0.002      -0.696      -0.161
    sulphates                0.8430      0.131      6.411      0.000       0.585       1.101
    alcohol                  0.2954      0.020     14.772      0.000       0.256       0.335
    ==============================================================================
    Omnibus:                       19.144   Durbin-Watson:                   2.022
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               26.903
    Skew:                          -0.181   Prob(JB):                     1.44e-06
    Kurtosis:                       3.668   Cond. No.                     1.58e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.58e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.


Awesome! Even after removing those four variables, we have an R-squared value of 0.987. We can also observe our result parameters here, and draw some insights regarding a white wine. "Importance" is a measure derived from the coefficient - the further from 0, the more important.

| **Variable** | **Coefficient** | **Insight** | **Importance** |
| ------------ | --------------- | ----------- | ---------------- |
| Volatile Acidity | -1.1082 | The **higher** your volatile acidity, the **lower** your quality. | Medium |
| Chlorides | -2.0665 | The **higher** your chlorides, the **lower** your quality. | High |
| Total Sulfure Dioxide | -0.0027 | The **higher** your total sulfur dioxide, the **lower** your quality. | Low |
| Density | 4.3390 | The **higher** your density, the **higher** your quality. | High |
| pH | -0.4287 | The **higher** your pH, the **lower** your quality. | Medium-Low |
| Sulphates | 0.8430 | The **higher** your sulphates, the **higher** your quality. | Medium-Low |
| Alcohol | 0.2954 | The **higher** your alcohol content, the **higher** your quality. | Medium-Low |

### White Wines

This process is essentially the same as our regression for red wine, but with our white wine data instead.


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
    Date:                Thu, 11 May 2017   Prob (F-statistic):               0.00
    Time:                        00:28:38   Log-Likelihood:                -3933.1
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


Hm, right off the bat we can see that the variables with a higher p-value are different for white whines than for red wines.

| **Item** | **P-Value** |
| ----- | ----- |
| Citric acid | 0.659 |
| Chlorides | 0.078 |
| Total sulfur dioxide | 0.316 |
| pH | 0.119 |


```python
# Optimize result
exemptions = ['color', 'quality', 'citric_acid', 'chlorides', 'total_sulfur_dioxide', 'pH']
newFormula = 'quality ~ 0 + ' + " + ".join([column for column in df if column not in exemptions])
result = ols(dfWhite, newFormula)
print result.summary()
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                quality   R-squared:                       0.983
    Model:                            OLS   Adj. R-squared:                  0.983
    Method:                 Least Squares   F-statistic:                 2.909e+04
    Date:                Thu, 11 May 2017   Prob (F-statistic):               0.00
    Time:                        00:28:42   Log-Likelihood:                -3936.7
    No. Observations:                3428   AIC:                             7887.
    Df Residuals:                    3421   BIC:                             7930.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    =======================================================================================
                              coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    fixed_acidity          -0.0693      0.016     -4.352      0.000      -0.101      -0.038
    volatile_acidity       -2.1379      0.130    -16.467      0.000      -2.392      -1.883
    residual_sugar          0.0268      0.003      8.972      0.000       0.021       0.033
    free_sulfur_dioxide     0.0032      0.001      3.928      0.000       0.002       0.005
    density                 2.4150      0.199     12.154      0.000       2.025       2.805
    sulphates               0.3771      0.115      3.271      0.001       0.151       0.603
    alcohol                 0.3879      0.012     32.236      0.000       0.364       0.412
    ==============================================================================
    Omnibus:                       86.141   Durbin-Watson:                   2.038
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              193.739
    Skew:                           0.080   Prob(JB):                     8.51e-43
    Kurtosis:                       4.153   Cond. No.                         652.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


Interesting. Our R-squared value actually went down by 0.001, but that's a good sign that we're not overfitting. Let's take a look at our white wine data.

| **Variable** | **Coefficient** | **Insight** | **Importance** |
| ------------ | --------------- | ----------- | ---------------- |
| Fixed Acidity | -0.0693 | The **higher** your fixed acidity, the **lower** your quality. | Low |
| Volatile Acidity | -2.1379 | The **higher** your volatile acidity, the **lower** your quality. | High |
| Residual Sugar | 0.0268 | The **higher** your residual sugar, the **higher** your quality. | Low |
| Free Sulfur Dioxide | 0.0032 | The **higher** your free sulfur dioxide, the **higher** your quality. | Low |
| Density | 2.4150 | The **higher** your density, the **higher** your quality. | High |
| Sulphates | 0.3771 | The **higher** your sulphates, the **higher** your quality. | Medium-Low |
| Alcohol | 0.3879 | The **higher** your alcohol content, the **higher** your quality. | Medium-Low |

### Insights

What are our key takeaways from this?
* Red wine makers should focus on **increasing density**, **decreasing volatile acidity**, and **decreasing chlorides**.
* White wine makers should focus on **increasing density** (but not the point of red wine), **decreasing volatile acidity**, and **increasing alcohol content**.
* Who likes white wine anyway?

# The rest of this article is a work in progress!

## Logistic Regression

```python
# Determine color from other traits

# Create formula to identify color
formula = "color ~ 0 + " + " + ".join([column for column in df if column != 'color'])
formula
```


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


```python
# What's important?
weights = Series(model.coef_[0], index=X.columns.values)
weights.sort_values()
```


```python
# Can we make a chart to figure out how great a wine is?
traitsToExclude = ['color', 'quality']
formula = 'C(quality) ~ 0 + C(color) + ' + " + ".join([column for column in df if column not in traitsToExclude])
formula
```


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


```python
tree.export_graphviz(model, feature_names=X.columns)
os.system('dot -Tpng tree.dot -o tree.png')
```

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


```python

```