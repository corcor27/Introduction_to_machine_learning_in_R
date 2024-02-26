---
title: "Non-Linear Classifiers"
teaching: 15
exercises: 20
questions:
- "How can I use scikit-learn to process data?"
objectives:
- "Recall that scikit-learn has built in linear regression functions."
- "Measure the error between a regression model and real data."
- "Apply scikit-learn's linear regression to create a model."
- "Analyse and assess the accuracy of a linear model using scikit-learn's metrics library."
- "Understand that more complex models can be built with non-linear equations."
- "Apply scikit-learn's polynomial modelling to non-linear data."
keypoints:
- "Scikit Learn is a Python library with lots of useful machine learning functions."
- "Scikit Learn includes a linear regression function."
- "It also includes a polynomial modelling function which is useful for modelling non-linear data."
---


## K-Nearest Neighbour (KNN)

The k-nearest Neighbours algorithm, commonly referred to as KNN or k-NN, is a supervised learning classifier that falls under the non-parametric category. It leverages proximity to classify or predict the grouping of a specific data point. Although it can tackle both regression and classification tasks, it is predominantly employed as a classification tool. The underlying principle is based on the assumption that similar data points tend to cluster together.
In classification scenarios, the algorithm assigns a class label through a majority vote mechanism. In other words, the label that appears most frequently among neighboring data points is adopted. While technically termed "plurality voting," it is often referred to as "majority vote" in literature. The distinction lies in the requirement for a true majority (over 50%), which suits binary classification situations. In cases involving multiple classes (e.g., four categories), a conclusive decision regarding a class label can be made with a threshold vote exceeding 25%.

~~~
library(caTools)

set.seed(1)
split = sample.split(iris$Sepal.Length, SplitRatio = 0.75)
train = subset(iris, split==TRUE)
test = subset(iris, split==FALSE)

train_scaled = scale(train[-5])
test_scaled = scale(test[-5])
~~~
{: .language-r}

~~~
library(class)
test_pred <- knn(train = train_scaled, test = test_scaled,cl = train$Species, k=10)

~~~
{: .language-r}

~~~
actual <- test$Species
cm <- table(actual,test_pred)
accuracy <- sum(diag(cm))/length(actual)
sprintf("Accuracy: %.f%%", accuracy*100)
~~~
{: .language-r}

## Support Vector Machines (KNN)

~~~
svm_model <- svm(Species ~ ., data=train_scaled, kernel="linear") #linear/polynomial/sigmoid

~~~
pred = predict(svm_model,iris)
tab = table(Predicted=pred, Actual = iris$Species)
tab
~~~
{: .language-r}

pred = predict(svm_model,test_scaled)
tab = table(Predicted=pred, Actual = iris$Species)
tab
~~~
{: .language-r}

> ## Exercise: Comparing linear and polynomial models
> Train a linear and polynomial model on life expectancy data from China between 1960 and 2000. Then predict life expectancy from 2001 to 2016 using both methods. Compare their root mean squared errors, which is more accurate? Why do you think this model is the more accurate one?
> > ## Solution
> > modify the call to the process_life_expectancy_data
> > ~~~
> > process_life_expectancy_data_poly("../data/gapminder-life-expectancy.csv", "China", 1960, 2000)
> > ~~~
> > {: .language-python}
> >
> > linear prediction error is  5.385162846665607
> > polynomial prediction error is 28.169167771983528
> > The linear model is more accurate, polynomial models often become wildly inaccurate beyond the range they were trained on. Look at the predicted life expectancies, the polynomial model predicts a life expectancy of 131 by 2016!
> > ![China 1960-2000](../fig/polynomial_china_training.png)
> > ![China 2001-2016 predictions](../fig/polynomial_china_overprediction.png)
> {: .solution}
{: .challenge}

{% include links.md %}
