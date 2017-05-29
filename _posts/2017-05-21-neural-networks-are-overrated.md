---
layout: post
published: true
title: Neural Networks are Overrated
blurb: Neural networks are overrated. Let's find out why.
tags:
    - python
    - machine learning
    - neural networks
---
     
Okay, okay, sorry for the clickbait title. I guess a more proper introduction to this article would be "Neural Networks **(for personal projects)** are Overrated". Before I took my first Machine Learning course at [The University of Texas at Austin](http://ianmobbs.com/), I attempted to learn various ML techniques on my own. I watched [Sirajology](https://www.youtube.com/channel/UCWN3xxRkmTPmbKwht9FuE5A) videos religiously. I tried retraining [Tensorflow's Inception-v3 network](https://www.tensorflow.org/tutorials/image_recognition). I skimmed through [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com) whenever I had spare time. I was looking for a more practical rather than theoretical introduction, which these resources did very well - but there was an issue. Each of these resources praised neural networks as the pinnacle of machine learning. While they may be a powerful tool, it's not appropriate to use them for every single problem. Newcomers to machine learning need to learn the basics. Unless you're working on a massive scale, tried and true algorithms (such as those found in [scikit-learn](http://scikit-learn.org/)) should almost always be used instead.

There's much more to the world of machine learning than neural networks, and assuming a neural network should be used to solve every problem is a dangerous mindset to have. [Here's an excellent StackOverflow discussion](http://stackoverflow.com/questions/1402370/when-to-use-genetic-algorithms-vs-when-to-use-neural-networks) on when to use Genetic Algorithms vs Neural Networks. If you want to read a little more about that, I suggest [checking here](http://bfy.tw/BvIt). In addition to what's been mentioned, neural networks are (by definition) the tool of choice for [deep learning](https://en.wikipedia.org/wiki/Deep_learning):

> Deep learning is a class of machine learning algorithms that use a cascade of many layers of nonlinear processing units for feature extraction and transformation.

For most personal projects though, a neural network isn't needed. I'll demonstrate that by analyzing the dataset [Homicide Reports, 1980-2014](https://www.kaggle.com/murderaccountability/homicide-reports) from the [Murder Accountability Project](https://www.kaggle.com/murderaccountability), who hosted their data on [Kaggle](https://kaggle.com/).

## The Data

The data contains 638,454 records of homicides across the country and the following information about each homicide:

* Record ID
* Agency Code
* Agency Name
* Agency Type
* City
* State
* Year
* Month
* Incident
* Crime Type
* Crime Solved
* Victim Sex
* Victim Age
* Victim Race
* Victim Ethnicity
* Perpetrator Sex
* Perpetrator Age
* Perpetrator Race
* Perpetrator Ethnicity
* Relationship
* Weapon
* Victim Count
* Perpetrator Count
* Record Source


```python
# Imports
import time
import pandas as pd
import numpy
from sklearn.metrics import accuracy_score

# Inline graphics 
%pylab inline

# Read data
df = pd.read_csv('database.csv')
df[:5]
```

    Populating the interactive namespace from numpy and matplotlib


    /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2683: DtypeWarning: Columns (16) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Record ID</th>
      <th>Agency Code</th>
      <th>Agency Name</th>
      <th>Agency Type</th>
      <th>City</th>
      <th>State</th>
      <th>Year</th>
      <th>Month</th>
      <th>Incident</th>
      <th>Crime Type</th>
      <th>...</th>
      <th>Victim Ethnicity</th>
      <th>Perpetrator Sex</th>
      <th>Perpetrator Age</th>
      <th>Perpetrator Race</th>
      <th>Perpetrator Ethnicity</th>
      <th>Relationship</th>
      <th>Weapon</th>
      <th>Victim Count</th>
      <th>Perpetrator Count</th>
      <th>Record Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>January</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>15</td>
      <td>Native American/Alaska Native</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Blunt Object</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>March</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>1</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Male</td>
      <td>42</td>
      <td>White</td>
      <td>Unknown</td>
      <td>Acquaintance</td>
      <td>Strangulation</td>
      <td>0</td>
      <td>0</td>
      <td>FBI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>AK00101</td>
      <td>Anchorage</td>
      <td>Municipal Police</td>
      <td>Anchorage</td>
      <td>Alaska</td>
      <td>1980</td>
      <td>April</td>
      <td>2</td>
      <td>Murder or Manslaughter</td>
      <td>...</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>Unknown</td>
      <td>0</td>
      <td>1</td>
      <td>FBI</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



## The Problem

In order to demonstrate that other machine learning techniques can be just as powerful as neural networks, without the cost, we're going to solving a simple classification problem using our data (each of these columns contains categorical data anyway, so it'd be pretty difficult to do any regression - but if anyone has a problem they'd like me to solve, I'd be happy to!). 

> For any crime, can we predict the race of the perpetrator based on other information?

Once you've identified your problems, the folks over at `scikit-learn` have created an [excellent cheatsheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/) on how to pick an algorithm to use:

![Pick an algorithm!](/assets/fnn/pick an algo.png)

Following the chart, we're going to use a [SGD Classifier](http://scikit-learn.org/stable/modules/sgd.html). In order to classify a perpetrators race based on other data, I decided to look at the dataframe columns by hand and use the features that I deemed important. **This is terrible practice**. I did this because, due to the sheer volume of data, training on every feature and then isolating what's important would've been far too difficult for my little computer. The point of this article is to highlight the difference in training times for neural networks and algorithms to reach similar results - not to create a perfect classifier.

## Data Sanitization

To clean up and split our data, we do a little bit of preprocessing on our own and let [patsy](http://patsy.readthedocs.io/en/latest/API-reference.html) take care of the rest. We change our dataset to only contain homicides where the perpetrators race **is** known and the crime **is** solved. This is because whether or not a crime is solved has no effect on the perpetrators race (although the opposite may be true), and we can't train on unknown data. After this, we write a formula using the isolated features, create some design matrices, and split our data into training and testing data.


```python
from patsy import dmatrices
from sklearn.model_selection import train_test_split

# Isolate data where perpetrators race is known and crime is solved
start = time.time()
data = df[(df['Perpetrator Race'] != 'Unknown') & (df['Crime Solved'] == 'Yes')] # Race known, case solved - Training data
end = time.time()
print("Time taken separating data:", end - start)

# Create patsy formula using different information
geographicInfo = "City + State"
crimeInfo = "Q('Crime Type') + Weapon + Incident"
victimInfo = "Q('Victim Sex') + Q('Victim Age') + Q('Victim Race') + Q('Victim Ethnicity') + Relationship"
formula = "Q('Perpetrator Race') ~ 0 + " + " + ".join([geographicInfo, crimeInfo, victimInfo])

# Split data into design matrices
start = time.time()
_, X = dmatrices(formula, data, return_type='dataframe')
y = data['Perpetrator Race']
end = time.time()
print("Time taken creating design matrices:", end - start)

# Split data into training and testing data
start = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
end = time.time()
print("Time taken splitting data:", end - start)

print("Total data size:", len(data))
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))
baseline = data['Perpetrator Race'].value_counts()[0] / data['Perpetrator Race'].value_counts().sum()
print("Baseline accuracy: ", baseline)
```

    Time taken separating data: 0.21260476112365723
    Time taken creating design matrices: 58.753602027893066
    Time taken splitting data: 5.484054088592529
    Total data size: 442123
    Training data size: 296222
    Testing data size: 145901
    Baseline accuracy:  0.493432822993


### SGD Classifier

Creating an SGD Classifier with `scitkit-learn` is incredible easy - as you can see, it takes three lines of code to instantiate, train, and predict with your classifier. It's performance is lacking (but again - our features were picked not to maximize accuracy, but to compare accuracy in tandem with training times). While 82% accuracy is low, it's a 33% accuracy improvement over our baseline. The 17 second training time on 296,000 rows of data is impressive.


```python
from sklearn import linear_model

start = time.time()
classifier = linear_model.SGDClassifier()
classifier.fit(X_train, y_train)
end = time.time()
print("SGDClassifier Training Time:", end - start)

start = time.time()
predictions = classifier.predict(X_test)
end = time.time()
print("SGDClassifier Prediction Time:", end - start)

print("SGDClassifier Accuracy:", accuracy_score(predictions, y_test))
```

    SGDClassifier Training Time: 17.512683868408203
    SGDClassifier Prediction Time: 0.7582650184631348
    SGDClassifier Accuracy: 0.827568008444


### Neural Network

In order for the neural network (which you can read about [here](http://scikit-learn.org/stable/modules/neural_networks_supervised.html#regression)) to train in a somewhat timely manner on my Macbook Pro, I had to stifle it's capabilities significantly by adjusting the size of it's hidden layers and the number of iterations. Even after these customizations though, training took 387 seconds - around **22.7** times longer than the SGDClassifer, with only a **4.5%** accuracy increase.


```python
from sklearn.neural_network import MLPClassifier

start = time.time()
classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)
classifier.fit(X_train, y_train)
end = time.time()
print("Neural Network Training Time:", end - start)

start = time.time()
predictions = classifier.predict(X_test)
end = time.time()
print("Neural Network Prediction Time:", end - start)

print("Neural Network Accuracy:", accuracy_score(predictions, y_test))
```

    Neural Network Training Time: 387.71303701400757
    Neural Network Prediction Time: 1.6371817588806152
    Neural Network Accuracy: 0.872084495651


## Conclusion

It's clear that the SGD Classifier outperformed the Neural Network. I hope this quick article goes to show that there's more to Machine Learning than neural networks, and that when solving an ML problem, all options should be considered. If you're looking for a practical introduction to Machine Learning, the book I used (that I highly recommend) is [Müeller](http://amueller.github.io/) and [Guido](http://www.oreilly.com/pub/au/6105)'s ["Introduction to Machine Learning with Python: A Guide for Data Scientists"](https://smile.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413/ref=sr_1_4?ie=UTF8&qid=1495839877&sr=8-4&keywords=machine+learning+python).

## Addendum

Andreas Mueller commented on the version of my article hosted on [The Practical Dev](https://dev.to/mobbsdev/neural-networks-are-overrated), and his comment was very insightful - I'd like to share it here:

> Cool article and an important point. Thanks for recommending our book! Neural networks are certainly not a cure-all, though it's tricky business to compare different algorithms on the same data because there are so many hyper parameters. For example increasing the number of iterations or using the sag sober solver might have improved the linear model, but take longer. Similarly a larger hidden layer (or changing any of the other tuning parameters) might have positive effects for the neural network.
> I think the main takeaway should be: never try neural networks first. Start with something simple and try complex models later if the gain in accuracy justifies for the added complexity in your application.

