---
title: "Non-Linear Classifiers"
teaching: 15
exercises: 20
questions:
- "How can I process data?"
objectives:
- "Recall how to build non-linear models."
- "Understand that more complex models can be built with non-linear equations."
- "To be able to predict using non-linear models"
keypoints:
- "Learning powerful library's to implement machine learning functions."
- "Used non-linear machine learning models to predict results"
---


## K-Nearest Neighbour (KNN)

The k-nearest Neighbours algorithm, commonly referred to as KNN or k-NN, is a supervised learning classifier that falls under the non-parametric category. It leverages proximity to classify or predict the grouping of a specific data point. Although it can tackle both regression and classification tasks, it is predominantly employed as a classification tool. The underlying principle is based on the assumption that similar data points tend to cluster together.
In classification scenarios, the algorithm assigns a class label through a majority vote mechanism. In other words, the label that appears most frequently among neighboring data points is adopted. While technically termed "plurality voting," it is often referred to as "majority vote" in literature. The distinction lies in the requirement for a true majority (over 50%), which suits binary classification situations. In cases involving multiple classes (e.g., four categories), a conclusive decision regarding a class label can be made with a threshold vote exceeding 25%.

Before we train any non-linear machine learning models, we need to divide our data into train and test sets. To do this we use a library called caTools. 
Furthermore, traditionally machine learning models only accept inputs which are between zero and one. so we will also need to scale our data.  

~~~
> library(caTools)

> set.seed(1)
> split = sample.split(iris$Sepal.Length, SplitRatio = 0.75)
> train = subset(iris, split==TRUE)
> test = subset(iris, split==FALSE)
> train_scaled = scale(train[-5])
> test_scaled = scale(test[-5])
> train_scaled
~~~
{: .language-r}

><pre style="color: black; background: white;">
>Sepal.Length  Sepal.Width Petal.Length  Petal.Width       setosa    virginica   versicolor 
>   5.8522124    3.0663717    3.6734513    1.1513274    0.3628319    0.3362832    0.3008850 
>attr(,"scaled:scale")
>Sepal.Length  Sepal.Width Petal.Length  Petal.Width       setosa    virginica   versicolor 
>   0.8523180    0.4524952    1.8304477    0.7617080    0.4829586    0.4745415    0.4606857 
></pre>
{: .output}

Now lets build our self KNN model, which we use a library called class.

~~~
> library(class)
> test_pred <- knn(train = train_scaled, test = test_scaled,cl = train$Species, k=2)
> test_pred
~~~
{: .language-r}

><pre style="color: black; background: white;">
> [1] setosa     setosa     setosa     setosa     setosa     setosa     setosa     setosa     setosa     versicolor versicolor
>[12] versicolor versicolor versicolor versicolor versicolor versicolor versicolor versicolor versicolor versicolor versicolor
>[23] versicolor versicolor versicolor virginica  virginica  virginica  virginica  virginica  virginica  virginica  virginica 
>[34] virginica  virginica  virginica  virginica 
>Levels: setosa versicolor virginica
></pre>
{: .output}

### Confusion Matrix

To look at how our model performed, there are a number of ways you could look at it. The best way is to have look at the confusion matrix and luckily in R there is a built in function that does this for us. All we have to do is pass our prediction results to the table function. Furthermore, by summing the diagonal and dividing by the length of our test set we can come up with an accuracy value. 

~~~
> actual <- test$Species
> cm <- table(actual,test_pred)
> cm
> accuracy <- sum(diag(cm))/length(actual)
> sprintf("Accuracy: %.f%%", accuracy*100)

~~~
{: .language-r}



~~~
"Accuracy: 92%"

           test_pred
actual       setosa versicolor virginica
setosa          9          0         0
versicolor      0         16         0
virginica       0          3         9
~~~
{: .output}

## Support Vector Machines (SVM)

The Support Vector Machine (SVM) emerges as a formidable supervised algorithm, demonstrating its effectiveness particularly on smaller yet intricate datasets. While adept at handling both regression and classification tasks, SVMs notably shine in classification scenarios. Originating in the 1990s, SVMs garnered widespread recognition and endure as a favoured option for high-performance algorithms, often requiring minimal adjustments to yield robust outcomes. Described as a machine learning algorithm utilising supervised learning models, SVMs tackle intricate classification, regression, and outlier detection challenges by executing optimal data transformations. These transformations delineate boundaries between data points based on predefined classes, labels, or outputs. This article elucidates the core principles of SVMs, their functionality, variations, and offers insights through real-world illustrations.

### Strengths of support vector machines:

- Effective in navigating high-dimensional spaces.
- Remain potent even when faced with a higher number of dimensions compared to samples.
- Operate efficiently on memory by utilizing a subset of training points known as support vectors in the decision-making process.
- Offer versatility through the option to specify various Kernel functions for the decision function, including the provision for custom kernels.

### Drawbacks of support vector machines:

- When the number of features significantly exceeds the number of samples, guarding against over-fitting necessitates careful selection of Kernel functions and regularization terms.
- Direct probability estimates are not provided by SVMs; obtaining such estimates involves resource-intensive techniques like five-fold cross-validation (refer to Scores and probabilities).

### SVM in R

So to create a SVM model, we are going to use the library called "e1071". We are also going to use our train/test separations from above.

~~~
> library(e1071)
> Species <- train$Species
> svm_model <- svm(Species ~ ., data=train_scaled, kernel="linear") #linear/polynomial/sigmoid
~~~
{: .language-r}

Now lets have ago at predicting our test set using the SVM model. Again we are going to produce a confusion matrix and generate an accuracy score.

~~~
> pred = predict(svm_model,test_scaled)
> tab = table(Predicted=pred, Actual = test$Species)
> tab
> accuracy <- sum(diag(tab))/length(test$Species)
> sprintf("Accuracy: %.f%%", accuracy*100)
~~~
{: .language-r}
~~~
"Accuracy: 92%"

            Actual
Predicted    setosa versicolor virginica
setosa          9          0         0
versicolor      0         16         3
virginica       0          0         9
~~~
{: .output}

> ## different non-linear classifier
>
> Have ago at implementing a different non-linear classifier. examples of decision tree can be found at: https://www.datacamp.com/tutorial/decision-trees-R
> Or even Random forest: https://www.r-bloggers.com/2021/04/random-forest-in-r/
{: .challenge}



{% include links.md %}
