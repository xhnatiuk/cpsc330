# Hyperparameter Optimization

In order to improve the generalization performance, finding the best values for the important hyperparameters of a model is necessary.

The problem of finding the best values for the important hyperparameters is tricky because

- You may have a lot of them (e.g. deep learning).
- You may have multiple hyperparameters which may interact with each other in unexpected ways.
- The best settings depend on the specific data/problem.

Manual or expert knowledge or heuristics based optimization

- Advantage: 
  * we may have some intuition about what might work.
  - E.g. if I’m massively overfitting, try decreasing `max_depth` or `C`.
- Disadvantages
  - it takes a lot of work
  - not reproducible
  - in very complicated cases, our intuition might be worse than a data-driven approach

Data-driven or automated optimization

- Formulate the hyperparamter optimization as a one big search problem. Often, the search space is quite big and systematic search for optimal values is infeasible.

- Advantages
  
  - reduce human effort
  - less prone to error and improve reproducibility
  - data-driven approaches may be effective

- Disadvantages
  
  - may be hard to incorporate intuition
  - be careful about overfitting on the validation set

We are going to talk about two such most commonly used automated optimizations methods from `scikit-learn`.

- Exhaustive grid search: [`sklearn.model_selection.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Randomized search: [`sklearn.model_selection.RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)

The “CV” stands for cross-validation; these methods have built-in cross-validation.

## Exhaustive Grid Search

For `GridSearchCV` we need

- an instantiated model or a pipeline
- a parameter grid: A user specifies a set of values for each hyperparameter.
- other optional arguments

The tricky part is we do not know in advance what range of hyperparameters might work the best for the given problem, model, and the dataset.

```python
from sklearn.model_selection import GridSearchCV

pipe_svm = make_pipeline(StandardScaler(), SVC())

param_grid = {
    "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10, 100],
    "svc__C": [0.001, 0.01, 0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(
    pipe_svm, param_grid, cv=5, n_jobs=-1, return_train_score=True
)

grid_search.fit(X_train, y_train) # all the work is done here
grid_search

grid_search.best_score_
grid_search.best_params_
```

It is often helpful to visualize results of all cross-validation experiments.

- You can access this information using `cv_results_` attribute of a fitted `GridSearchCV` object.

Other than searching for best hyperparameter values, `GridSearchCV` also fits a new model on the whole training set with the parameters that yielded the best results. So, we can conveniently call `score` on the test set with a fitted `GridSearchCV` object.

```python
grid_search.score(X_test, y_test)
```

Note the `n_jobs=-1` above.

- Hyperparameter optimization can be done *in parallel* for each of the configurations.
- This is very useful when scaling up to large numbers of machines in the cloud.

### The `__` syntax

- Above: we have a nesting of transformers.
- We can access the parameters of the “inner” objects by using __ to go “deeper”:
- `svc__gamma`: the `gamma` of the `svc` of the pipeline
- `svc__C`: the `C` of the `svc` of the pipeline

### Issues

Required number of models to evaluate grows exponentially with the dimensionally of the configuration space.

- 5 hyperparameters
- 10 different values for each hyperparameter
- You’ll be evaluating $10^5$ models! That is you’ll be calling `cross_validate` 100,000 times!
- Exhaustive search may become infeasible fairly quickly.

## Randomized Grid Search

Randomized hyperparameter optimization. Samples configurations at random until certain budget (e.g., time) is exhausted.

### `n_iter`

- Note the `n_iter`, we didn’t need this for `GridSearchCV`.
- Larger `n_iter` will take longer but it’ll do more searching.
  - Remember you still need to multiply by number of folds!

### Advantages

Faster compared to `GridSearchCV`.

- Adding parameters that do not influence the performance does not affect efficiency.
- Works better when some parameters are more important than others.
- In general reccomended over `GridSearchCV`.

## Optimization Bias (Overfitting of the Validation Set)

While carrying out hyperparameter optimization, we usually try over many possibilities. If our dataset is small and if your validation set is hit too many times, we suffer from **optimization bias** or **overfitting the validation set**.

![img](https://amueller.github.io/COMS4995-s20/slides/aml-03-supervised-learning/images/overfitting_validation_set_2.png)

Optimization Bias of Parameter Learning: Overfitting of the training error.

- During training, we could search over tons of different decision trees.
- So we can get “lucky” and find one with low training error by chance.

Optimization Bias of Hyperparameter Learning: Overfitting of the validation error.

- Here, we might optimize the validation error over 1000 values of `max_depth`.
- One of the 1000 trees might have low validation error by chance.
- Thus, not only can we not trust the cv scores, we also cannot trust cv’s ability to choose of the best hyperparameters.

This is why we need a test set. The frustrating part is that if our dataset is small then our test set is also small 😔.

If your test score is much lower than your CV score:

- Try simpler models and use the test set a couple of times
- Communicate clearly when you report your results

Large datasets solve many problems.

- This infinite training data overfitting would not be a problem. Theoretically you could have test score = train score.
- Overfitting happens because you only see a bit of data and you learn patterns that are overly specific to the sample. If you could see all the data the notion of overly specific would not apply.
- More data will make our test scores better and more robust

# Classification Metrics

If we have many examples of one class, then even a dummy classifier will score well on the dataset. This is called a class imbalance.

- Is accuracy a good metric here?
- Can we use something other than accuracy to compare our models?

## Confusion Matrix

One way to get a better understanding of the errors is by looking at:

- false positives (type I errors), where the model incorrectly spots examples as fraud
- false negatives (type II errors), where it’s missing to spot fraud examples

```python
from sklearn.metrics import confusion_matrix

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression())
pipe_lr.fit(X_train, y_train)
predictions = pipe_lr.predict(X_valid)
TN, FP, FN, TP = confusion_matrix(y_valid, predictions).ravel()
print(disp.confusion_matrix)
```

### What are Positive and Negative?

Two kinds of binary classification problems:

1. Distinguishing between two classes
2. Spotting a class (spot fraud transaction, spot spam, spot disease)

In case of spotting problems, the thing that we are interested in spotting is considered “positive”.

Note that what you consider as positive is important, if you flip what is considered to be positive and what is to be considered negative, we will end up with different TP, FP, TN, FN, and therefore different precision , recall, and f1 scores.

## Precision, Recall, F1 Score

We have been using `.score` to assess our models, which returns accuracy by default.

- Accuracy is misleading when we have class imbalance.
- We need other metrics to assess our models.

We’ll discuss three commonly used metrics which are based on confusion matrix:

1. recall
2. precision
3. f1 score

Note that these metrics will only help us assessing our model.`scikit-learn` has functions for [these metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).

### Recall

Among all positive examples, how many did you identify?

$$
recall = \frac{TruePositives}{TruePositives + FalseNegatives} = \frac{TruePositives}{AllRealPositives}
$$

### Precision

Among the positive examples you identified, how many were actually positive?

$$
precision = \frac{TruePositives}{TruePositives + FalsePositives} = \frac{TruePositives}{TotalPredictedPositives}
$$

### F1 Score

F1-score combines precision and recall to give one score, which could be used in hyperparameter optimization.

- F1-score is a harmonic mean of precision and recall.

$$
f1 = 2*\frac{precision * recall}{precision + recall}
$$

### Classification Report

There is a convenient function called `classification_report` in `sklearn` which gives this info.

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_valid, 
        pipe_lr.predict(X_valid),
        target_names=["non-fraud", "fraud"]
    )
)
```

Output:

```
              precision    recall  f1-score   support

   non-fraud       1.00      1.00      1.00     59708
       fraud       0.89      0.63      0.74       102

    accuracy                           1.00     59810
   macro avg       0.94      0.81      0.87     59810
weighted avg       1.00      1.00      1.00     59810
```

Which one is relevant when depends upon whether you think each class should have the same weight or each sample should have the same weight.

#### Macro Average

You give equal importance to all classes and average over all classes.

- For instance, in the example above, recall for non-fraud is 1.0 and fraud is 0.63, and so macro average is 0.81.
- More relevant in case of multi-class problems.

#### Weighted Average

You give equal importance to each example.

- Weighted by the number of samples in each class.
- Divide by the total number of samples.

### Cross Validation

We can pass different evaluation metrics with `scoring` argument of `cross_validate`.

```python
scoring = [
    "accuracy",
    "f1",
    "recall",
    "precision",
]  # scoring can be a string, a list, or a dictionary
pipe = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_validate(
    pipe, X_train_big, y_train_big, return_train_score=True, scoring=scoring
)
pd.DataFrame(scores)
```

## Precision-Recall Curve and ROC Curve

Confusion matrix provides a detailed break down of the errors made by the model. But when creating a confusion matrix, we are using “hard” predictions.

- Can we explore the degree of uncertainty to understand and improve the model performance?

**Key idea: what if we threshold the probability at a smaller value so that we identify more examples as “positive” examples?**

**Operating point**: Setting a requirement on a classifier (e.g., recall of >= 0.75) is called setting the **operating point**.

- It’s usually driven by business goals and is useful to make performance guarantees to customers.

### Precision/Recall Tradeoff

There is a trade-off between precision and recall. If you identify more things as “positive”, recall is going to increase but there are likely to be more false positives.

**Increasing the threshold**: higher bar for predicting positives.

- recall would go down or stay the same but precision is likely to go up
- occasionally, precision may go down as the denominator for precision is TP+FP.

**Decreasing the threshold**: lower bar for predicting positives.

- You are willing to risk more false positives in exchange of more true positives.
- recall would either stay the same or go up 
- precision is likely to go down
- occasionally, precision may increase if all the new examples after decreasing the threshold are TPs.

Remember to pick the desired threshold based on the results on the validation set and **not** on the test set.

### Precision/Recall Curve

Often, when developing a model, it’s not always clear what the operating point will be and to understand the the model better, it’s informative to look at all possible thresholds and corresponding trade-offs of precision and recall in a plot.

Often it’s useful to have one number summarizing the PR plot (e.g., in hyperparameter optimization)

- One way to do this is by computing the area under the PR curve.
- This is called **average precision** (AP score)
- AP score has a value between 0 (worst) and 1 (best).

```python
from sklearn.metrics import average_precision_score

ap_lr = average_precision_score(y_valid, pipe_lr.predict_proba(X_valid)[:, 1])
print("Average precision of logistic regression: {:.3f}".format(ap_lr))
```

#### AP vs. F1-Score

It is very important to note this distinction:

- F1 score is for a given threshold and measures the quality of `predict`.
- AP score is a summary across thresholds and measures the quality of `predict_proba`.

### Receiver Operating Characteristic (ROC) curve

Another commonly used tool to analyze the behavior of classifiers at different thresholds.

Similar to PR curve, it considers all possible thresholds for a given classifier given by `predict_proba` but instead of precision and recall it plots false positive rate (FPR) and true positive rate (TPR or recall).

$$
FPR = \frac{FalsePositives}{FalsePositives + TrueNegatives}
$$

$

$

$$
TPR = \frac{TruePositives}{TruePositives + FalsePositives}
$$

#### Area under the curve (AUC)

AUC provides a single meaningful number for the model performance.

- AUC of 0.5 means random chance.
- AUC can be interpreted as evaluating the **ranking** of positive examples.
- What’s the probability that a randomly picked positive point has a higher score according to the classifier than a randomly picked point from the negative class.
- AUC of 1.0 means all positive points have a higher score than all negative points.

For classification problems with imbalanced classes, using AP score or AUC is often much more meaningful than using accuracy.

## Dealing with Class Imbalance

A very important question to ask yourself: “Why do I have a class imbalance?”

- Is it because one class is much more rare than the other?
  - If it’s just because one is more rare than the other, you need to ask whether you care about one type of error more than the other.
  - We need to address class imbalance
- Is it because of my data collection methods?
  - If it’s the data collection, then that means *your test and training data come from different distributions*!
  - We need to address class imbalance

In some cases, it may be fine to just ignore the class imbalance.

### Handling imbalance

Depending on which kind of error is more important, we can pick a threshold that is appropriate for our problem.

Can we change the model itself rather than changing the threshold so that it takes into account the errors that are important to us?

There are two common approaches for this:

- **Changing the data (optional)** (not covered in this course)
  - Undersampling
  - Oversampling
    - Random oversampling
    - SMOTE
- **Changing the training procedure**
  - `class_weight`

### Changing the training procedure

- All `sklearn` classifiers have a parameter called `class_weight`.
- This allows you to specify that one class is more important than another.
- For example, maybe a false negative is 10x more problematic than a false positive.
- Changing the class weight will **generally reduce accuracy**.
  - The original model was trying to maximize accuracy.
  - Now you’re telling it to do something different.
  - But that can be fine, accuracy isn’t the only metric that matters.

A useful setting is `class_weight="balanced"`.

- This sets the weights so that the classes are “equal”.

### Stratified Splits

A similar idea of “balancing” classes can be applied to data splits.

- We have the same option in `train_test_split` with the `stratify` argument.

- By default it splits the data so that if we have 10% negative examples in total, then each split will have 10% negative examples.

- If you are carrying out cross validation using `cross_validate`, by default it uses [`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html). From the documentation:

> This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.

- In other words, if we have 10% negative examples in total, then each fold will have 10% negative examples.

Is this a good idea?:

- Well, it’s no longer a random sample, which is probably theoretically bad, but not that big of a deal.
- If you have many examples, it shouldn’t matter as much.
- It can be especially useful in multi-class, say if you have one class with very few cases.
- In general, these are difficult questions.

# Regression Metrics

We aren’t doing classification anymore, so we can’t just check for equality. We need a score that reflects how right/wrong each prediction is.

A number of popular scoring functions for regression. We are going to look at some common metrics:

- mean squared error (MSE)
- $R^2$
- root mean squared error (RMSE)
- MAPE

## Mean Squared Error (MSE)

Unlike classification, with regression **our target has units**. The score also depends on the scale of the targets.

- ex: If we were working in cents instead of dollars, our MSE would be be 10,000 times higher ($100^2$)

### Root Mean Squared Error (RMSE)

A more relatable metric would be the root mean squared error, or RMSE.

- Unfortunately outliers throw both MSE and RMSE way off

## $R^2$

This is the score that `sklearn` uses by default when you call score():

- similar to mean squared error, but flipped (higher is better)
- normalized so the max is 1.
- Negative values are very bad: “worse than `DummyRegressor`”

## MAPE

How about looking at percent error?

```python
pred_train = lr_tuned.predict(X_train)
percent_errors = (pred_train - y_train) / y_train * 100.0
np.abs(percent_errors) # absolute percent error

def mape(true, pred):
    return 100.0 * np.mean(np.abs((pred - true) / true))
```

Like MSE, we can take the average over examples. This is called mean absolute percent error (MAPE).

- this is quite interpretable.

- On average, we have around x% error.

# Feature Importances

## Interpretability

The ability to interpret our models is crucial in many applications such as: banking, healthcare, and criminal justice. It can be leveraged by domain experts to diagnose systematic errors and underlying biases of complex ML systems.

In this course our definition of model interpretability is **feature importance**. There are more factors in interpretability but this is a good start.

Feature importance does not have a sign!

- Only tells us about importance, nothing about up or down.

## Feature Correlations

One way for determining the importance of features is to look at the correlations between features and other features in our data.

- Positive Correlation: Y goes up when X goes up
- Negative Correlation: Y goes down when X goes up
- Uncorrelated: Y doesn't change when X goes up

This approach is extremely simplistic.

- It only looks at each feature in isolation
- It only looks at linear associations

Sometimes a feature only becomes important if another feature is *added* or *removed*.

### Ordinal Features

Ordinal features are the easiest to interpret. In a linear regression, if we increase our ordinal feature by 1 category, it effects our model by 1 times its learned coefficient.

### Categorical Features

With categorical features we consider one of the categories for a feature to be the reference category. We then calculate the difference between the other categories for the feature and the reference category to interpret their effect on the model.

- Ex: if feature 1 changed from category A (reference) to category B (non-reference) if would effect our model by its difference.
- Do we really believe these interpretations?
  - This is how predictions are being made, so yes
  - But this is likely not how the world works, so no

### Numeric Features

This is trickier than you would expect since we have scaled our numeric features. Our intuition should be that if we increase a numeric feature by 1 scaled unit then if effects our model by 1 times its learned coefficient.

To interpret a feature we should divide the learned coefficient from the model by the scale coefficient.

## Shapley Additive Explanations (SHAP)

A sophisticated measure of the contribution of each feature. We will not go into details of how it works, but we will learn how to use it.

We can use SHAP to explain predictions on our deployment data.

We can use average SHAP values to determine global feature importance.

Smaller SHAP values mean that we are less likely to get placed in the target class we are looking at. Synonymous with correlation?

# Feature Engineering

Better features -> more flexibility, higher score. We can get simple and interpretable models.

If your features i.e. representation is bad, whatever model you build is not going to help.

> Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.
> 
> -Jason Brownlee

Better features usually help more than a better model. Good features ideally:

- capture most important aspects of the problem
- allow learning with few examples
- generalize to new scenarrios

There is a trade-off between simple and expressive features

- simple features have a low overfitting risk, but scores might be low
- more complicated features scores can be high, but overfitting is a risk

In some domains there are natural transformations to do:

- Feature engineering is super domain specific so it is hard to provide generalized knowledge

The best feautres are application dependent. It is difficult to give general advice, some guidelines are to: 

* Ask domain experts

* Read academic papers in the domain

## Feature Selection

With so many ways to add new features, we can easily increase the dimensionality of our data. More features means more complex models, which also means a higher risk of overfitting. 

Feature Selection is finding the features that are important for predicting our target and removing the others. 

* **Increases interpretability**: no reason to use more features if it doesn't improve our performance.

* **Computation**: models fit/predict faster with fewer columns

* **Data collection**: may be cheaper with fewer features

### How do we select features?

We can use domain knowledge to discard features. 

We are going to look briefly at three automatic feature selection methods in `sklearn`, these are related to looking at feature importance:

- Model-based selection
- Recursive feature elimination
- Forward selection

#### Model-based selection

Uses a supervised machine learning model to judge the importance of our feature, and only keep the most important ones. 

#### Recursive feature elimination (RFE)

Build a series of models, at each iteration, discard the least important feature according to the model.

- Computationally expenseive

REF algorithm:

1. Decide k, the number of features to select
2. Assign importance to features (i.e. fit and remove lowest magnitude coefficient in regression)
3. Remove least important feature
4. Repeat until only k features are remaining

This is NOT the same as removing the k least important features all at once

How do we know what value to pass to `n_features_to_select`?

- Use `RFECV` which uses cross-validation to select number of features.

#### Search and Score

General idea of search and score methods:

- Define a scoring function that measures the quality of a set of features
- We try each member of the powerset of our features in an exhaustive search
- Too computationally expensive

#### Forward or Backward Selection

- Also called wrapper methods
- Shrink or grow feature set by removing or adding one feature at a time
- Makes the decision based on whether adding/removing the feature improves the CV score or not

#### Other ways to search

Stochastic local serach

* Inject randomness so that we can explore new parts of the serach space

* Simulated annealing

* Genetic algorithms

### Warnings regarding feature selection

A features relevance is only defined in the context of other features

- Adding/removing features can make features relevant/irrelevant

If features can be predicted from other features, you cannot know which one to pick

- Relevance for features does not have to be a causal relationship

The methods we have looked at do not discover new truths about how the world works, rather they just inform you which features are predicting the targets in your data. 

# EDA

`df.head()`: shows us some sample rows of our dataframe.

`df.info()`: gives us the column number, name, count, and datatype for our features. Note that strings will be datatype object.

`df.describe()`: gives us the count, mean, std, min, max, and more information about each feature.

The following code can be used to count the different values that occur in our target:`y.value_counts()`

We can use `.shape` to check the shape of our data splits.
