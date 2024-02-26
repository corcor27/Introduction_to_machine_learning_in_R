---
title: "Regression"
teaching: 45
exercises: 30
questions:
- "How can I make linear regression models from data?"
- "How can I use logarithmic regression to work with non-linear data?"
objectives:
- "Learn how to use linear regression to produce a model from data."
- "Learn how to model non-linear data using a logarithmic."
- "Learn how to measure the error between the original data and a linear model."
keypoints:
- "We can model linear data using a linear or least squares regression."
- "A linear regression model can be used to predict future values."
- "We should split up our training dataset and use part of it to test the model."
- "For non-linear data we can use logarithms to make the data linear."
---

# Linear regression

We now possess a basic linear model for a given dataset. It would be valuable to assess the accuracy of this model. One way to achieve this is by computing the predicted y-values for each x-value in our original dataset and comparing them with the actual y-values. We can aggregate these individual discrepancies into a single comprehensive error metric by calculating the least squares. This involves squaring each difference, summing them all, dividing the sum by the total number of observations, and then taking the square root of the result. By squaring and subsequently taking the square root, we prevent negative errors from offsetting positive ones, thus providing us with an overall error metric to gauge the accuracy of our model.


## Preprocess the dataset

Lets test our code by using the example data from the mathsisfun link above.

~~~
Y<- iris[,"Sepal.Width"] # select Target attribute
X<- iris[,"Sepal.Length"] # select Predictor attribute
head(X)
~~~
{: .language-r}

~~~
xycorr<- cor(Y,X, method="pearson") # find pearson correlation coefficient
xycorr # a value near 1 implies high correlation and that near 0 shows low correlation
~~~
{: .language-r}

~~~
plot(Y~X, col=X)
model1<- lm(Y~X)
model1 # provides regression line coefficients i.e. slope and y-intercept
~~~
{: .language-r}

~~~
plot(Y~X, col=X) # scatter plot between X and Y
abline(model1, col="blue", lwd=3) # add regression line to scatter plot to see relationship between X and Y
~~~
{: .language-r}

### Testing the accuracy of a linear regression model

Now, let’s use the line coefficients for two equations that we got in model1 and model2 to predict value of Target for any given value of Predictor.

~~~
# Prediction of 'Sepal.Width' when 'Sepal.Length'= 20
p1<- predict(model1,data.frame("X"=20))
p1
# Prediction of 'Petal.Width' when 'Petal.Length'= 15
p2<- predict(model2,data.frame("V"=15))
p2
~~~
{: .language-python}


> ## Comparing the logarithmic and non-logarithmic graphs
>
> Convert the code above to plot the logarithmic version of the graph.
> Save the graph.
> Now change back to the non-logarithmic version.
> Compare the two graphs, which one do you think is easier to read?
{: .challenge}


> ## Removing outliers from the data
> The correlation of GDP and life expectancy has a few big outliers that are probably increasing the error rate on this model. These are typically countries with very high GDP and sometimes not very high life expectancy. These tend to be either small countries with artificially high GDPs such as Monaco and Luxemborg or oil rich countries such as Qatar or Brunei. Kuwait, Qatar and Brunei have already been removed from this data set, but are available in the file worldbank-gdp-outliers.csv. Try experimenting with adding and removing some of these high income countries to see what effect it has on your model's error rate.
> Do you think its a good idea to remove these outliers from your model?
> How might you do this automatically?
{: .challenge}

{% include links.md %}
