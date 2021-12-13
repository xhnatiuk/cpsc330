Steps to train a classifier:

1. Read the data: 
2. Create X (feature vectors) and y (target)
3. Create a classifier object
4. `fit` the classifier
5. `predict` on new examples
6. `score` the model

```python
df = pd.read_csv("./my_data.csv")
X = df.drop(columns=["target"])
y = df["target"]
model = DummyClassifier(strategy="most_frequent")
model.fit(X, y)
# this step is somewhat optional, only use if you need to see predictions
predicitons = model.predict(X)
model.score(X, y)
```

# Baselines

**Baseline:** a simple machine learning algorithm based on simple rules of thumb. Provides a point of reference for comparing your machine learning model.

`sklearn` models: 

* `DummyClassifier`: predicts the most frequent label in the training set.

* `DummyRegressor` predicts mean, median, or a constant value of the training set.

# Decision Tree

Decision trees predict targest using a rule-based algorithim. They are similar to a number of conditional statements. They can be used for classifcation or regression.

* `DescisionTreeClassifier`

* `DecisionTreeRegressor`

Our goal is to minimize **impurity** at each question. Impurity is measured using:

* Classification: use gini index, but can also use information gain and cross entropy.

* Regression: mean squared error (MSE)

Scoring is done using:

* Classification: accuracy

* Regression: $R^2$ score
  
  * The maximum is 1 for perfect predictions
  
  * Can be negative, which is very bad (worse than `DummyRegressor`)

**Decision Stump**: a decision tree with only one split (`max_depth=1`).

### Parameters

The decision tree learns two things:

- the best feature to split on
- the threshold for the feature to split on at each node

Features can be binary, categorical, or continuous.

## Hyperparameters

Default ensures all leaf nodes are pure. 

- When a leaf node has no impurity, all of the data that reaches it will be labeled as one class.
- If a leaf node is not pure, it will just assign the majority class at that particular node.

We can control this using the `max_depth` hyperparameter, which is the length of the longest path from tree root to a leaf.

Some other commonly used hyperparameters of a decision tree are:

- `min_samples_split`
- `min_samples_leaf`
- `max_leaf_nodes`

# KNNs

KNNs are an analogy based model. They use euclidean distance to determine the k "closest" feature vectors to our test example. 

* Euclidean Distance between to vectors u and v is defined as:
  $distance(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}$

Can also be used for regression

- Take the average of the k-nearest neighbors
- We can also have weighted regression

`sklearn` models:

- `KNeighborsClassifier`

- `KNeighborsRegressor`

## Hyperparameters

 `n_neighbors` (k) defines how many neighbors should vote during predictions.

- as k increases we underfit our model; low values of k overfit our model

Other useful hyperparameters include `weights` which allows you to assign a higher weight to the examples which are closer to the vector in question.

### Pros

- Easy to understand and interpret
- Simple hyperparameter k controlling the fundamental tradeoff
- Can learn very complex functions if given enough data
- Lazy learning: takes no time to `fit` (just stores all the examples)

### Cons

- Can be potentially be VERY slow during prediction time, especially when the training set is very large. 
  - Has to calculate distances to every single example.
- Often not that great test accuracy compared to the modern approaches.
- It does not work well on datasets with many features (curse of dimensionality) or where most feature values are 0 most of the time (sparse datasets).

# Support Vector Machines (SVM) with RBF Kernel

Another popular similarity based algorithm is SVM RBFs. SVM RBFs are more like weighted k-NNs. SVMs can also be used for regression problems.

* RBF stands for radial basis functions. We will not learn about this.

**Main insight**: the decision boundary only depends on the support vectors.

We can think of SVM with RBF kernel as KNN with a smooth decision boundary.

- The decision boundary is defined by **a set of positive and negative examples** and **their weights** together with **their similarity measure**.
- A test example is labeled positive if on average it looks more like positive examples than the negative examples.
- Each training example either is or isn’t a “support vector”. This gets decided during `fit`
- We can examine the support vectors for the model using `my_model.support_`

The primary difference between k-NNs and SVM RBFs is that

- Unlike k-NNs, SVM RBFs only remember the key examples (support vectors). So it’s more efficient than k-NN.

- SVMs use a different similarity metric which is called a “kernel” in SVM land. A popular kernel is Radial Basis Functions (RBFs)

- They usually perform better than k-NNs!

`sklearn` models:

- Classifer: `SVC`

- Regressor: `SVR`

### Hyperparameters

`gamma` controls the complexity (fundamental trade-off), just like other hyperparameters we’ve seen.

- larger `gamma` means more complex

- smaller `gamma` means less complex

`C` also affects the fundamental tradeoff

- larger `C` means more complex

- smaller `C` means less complex

# Linear Models

**Linear models** are a fundamental and widely used class of models. They are called **linear** because they make a prediction using a **linear function** of the input features.

We will talk about three linear models:

1. Linear regression
2. Logistic regression
3. Linear SVM (brief mention)

Strengths of Linear Models:

- Fast to train and predict
- Scale to large datasets and work well with sparse data
- Relatively easy to understand and interpret the predictions
- Perform well when there is a large number of features

Limitations:

- Is your data “linearly separable”? Can you draw a hyperplane between these datapoints that separates them with 0 error.
- If the training examples can be separated by a linear decision rule, they are **linearly separable**.

### Scaling and Interpretability

When you are interpreting model coefficients, scaling is crucial.

- If you do not scale the data, features with smaller magnitude are going to get coefficients with bigger magnitude whereas features with bigger scale are going to get coefficients with smaller magnitude.
- That said, when you scale the data, feature values become hard to interpret for humans!

## Linear Regression

The model is a line, which can be represented with a slope (i.e., coefficient or weight) and an intercept.

In linear models for regression, the model is a line for a single feature, a plane for two features, and a hyperplane for higher dimensions.

When we call `fit`, a coefficient or weight is learned for each feature which tells us the role of that feature in prediction. These coefficients are learned from the training data.

For the model, we can access the slope (i.e., coefficient or weight) and the intercept using `coef_` and `intercept_`, respectively.

Given a feature x1, a learned coefficient w1 and an intercept b, we can get the prediction y hat with the following formula: $\hat{y} = w_1x_1 + b$

For more features, the model is a higher dimensional hyperplane and the general prediction formula looks as follows: $\hat{y} = w_1x_1 + ... + w_dx_d + b$ where x are our input features, w are our coefficients (weights learned from the data), and b is the bias which is used to offset our hyperplane (learned from the data).

`scikit-learn` has a model called `LinearRegression` for linear regression.

- But if we use this “vanilla” version of linear regression, it may result in large coefficients and unexpected results.

```python
from sklearn.linear_model import LinearRegression  # DO NOT USE IT
from sklearn.linear_model import Ridge  # USE THIS INSTEAD
```

### Ridge

Instead of using `LinearRegression`, we will always use another linear model called `Ridge`, which is a linear regression model with a complexity hyperparameter `alpha`.

#### Parameters

The model learns

- coefficients associated with each feature
- the intercept or bias
  - For each prediction, we are adding this amount irrespective of the feature values.

#### Hyperparameter `alpha` of `Ridge`

The alpha hyperparameter is what makes `Ridge` different from vanilla `LinearRegression`. Similar to the other hyperparameters that we saw, `alpha` controls the fundamental tradeoff.

- If we set alpha=0 that is the same as using `LinearRegression`.

- larger `alpha` means likely underfit

- smaller `alpha` likely overfit

### Interpretation of Coefficients

One of the main advantages of linear models is that they are relatively easy to interpret. We have one coefficient per feature which kind of describes the role of the feature in the prediction according to the model.

There are two pieces of information in the coefficients based on

- Sign
  - positive: the prediction will be proportional to the feature value; as it gets bigger our predicted value gets bigger
  - negative: the prediction will be inversely proportional to the feature value; as it gets bigger our predicted value gets smaller
- Magnitude
  - larger magnitudes have a bigger impact on our predictions

Take these coefficients with a grain of salt. They might not always match your intuitions.

## Logistic Regression

A linear model for **classification**. Similar to linear regression, it learns weights associated with each feature and the bias.

- It applies a **threshold** on the raw output to decide whether the class is positive or negative.
- We will focus on the following aspects of logistic regression.
  - `predict`, `predict_proba`
  - how to use learned coefficients to interpret the model

Uses learned coefficients and a learned bias similar to linear regression.

- the prediction is based on the weighted sum of the input features.

- Some features are pulling the prediction towards positive and some are pulling it towards negative.

The components of a linear classifier are:

1. input features (x1, ..., xd)
2. learned coefficients (weights) (w1, ..., wd)
3. bias (b or w0) used to offset the hyperplane
4. threshold (r) used for classification

### Parameters

Similar to `Ridge`, we can access the weights and intercept using `coef_` and `intercept_` attribute of the `LogisticRegression` object, respectively.

With logistic regression, the model randomly assigns one of the classes as a positive class and the other as negative.

- Usually it would alphabetically order the target and pick the first one as negative and second one as the positive class.

- The `classes_` attribute tells us which class is considered negative and which one is considered

The decision boundary of logistic regression is a **hyperplane** dividing the feature space in half.

- For d=2, the decision boundary is a line (1-dimensional)
- For d=3, the decision boundary is a plane (2-dimensional)
- For d>3, the decision boundary is a (d-1)-dimensional hyperplane

### Hyperparameters

`C` is the main hyperparameter which controls the fundamental trade-off. At a high level, the interpretation is similar to `C` of SVM RBF

- smaller `C` might lead to underfitting

- bigger `C` might lead to overfitting

### Probability Scores

So far in the context of classification problems, we focused on getting “hard” predictions. Very often it’s useful to know “soft” predictions, i.e., how confident the model is with a given prediction.

For most of the `scikit-learn` classification models we can access this confidence score or probability score using a method called `predict_proba`.

* The output of `predict_proba` is the probability of each class.

* The first entry is the estimated probability of the first class and the second entry is the estimated probability of the second class from `model.classes_`.

* Because it’s a probability, the sum of the entries for both classes should always sum to 1.

* Since the probabilities for the two classes sum to 1, exactly one of the classes will have a score >=0.5, which is going to be our predicted class.

#### Calculating Probabilities

The linear regression equation gives us our "raw model output". In a linear regression this is our prediction. In logistic regression we check the sign of this value.

- If positive (or 0), predict 1; if negative, predict -1
- These are “hard predictions”.

We can also have "soft predictions", our predicted probabilities. To convert the raw model output into probabilities, instead of just taking the sign, we need to apply the sigmoid function.

The sigmoid function “squashes” the raw model output from any number to the range [0,1] using the following formula, where x is the raw model output.

$$
\frac{1}{1 + e^{-x}}
$$

Then we can interpret the output as probabilities.

#### Confidence

Sometimes a complex model that is overfitted, tends to make more confident predictions, even if they are wrong, whereas a simpler model tends to make predictions with more uncertainty.

- With hard predictions, we only know the class.
- With probability scores we know how confident the model is with certain predictions, which can be useful in understanding the model better.

## Linear SVM

There is also a linear SVM. You can pass `kernel="linear"` to create a linear SVM.

- `predict` method of linear SVM and logistic regression works the same way.

- We can get `coef_` associated with the features and `intercept_` using a Linear SVM model.

Note that the coefficients and intercept are slightly different for logistic regression. This is because the `fit` for linear SVM and logistic regression are different.

# Ensembles

**Ensembles** are models that combine multiple machine learning models to create more powerful models.

Decision trees models are

- Interpretable
- They can capture non-linear relationships
- They don’t require scaling of the data and theoretically can work with categorical features.

But with a single decision, trees are likely to overfit.

**Key idea**: Combine multiple trees to build stronger models.

- These kinds of models are extremely popular in industry and machine learning competitions

## Random Forests

Use a collection of diverse decision trees. Each tree overfits on some part of the data but we can reduce overfitting by averaging the results.

- can be shown mathematically

How?:

1. Decide how many decision trees we want to build
   - can control with `n_estimators` hyperparameter
2. `fit` a diverse set of that many decision trees by **injecting randomness** in the classifier construction
3. `predict` by voting (classification) or averaging (regression) of predictions given by individual models

What does it do?:

1. Create a collection (ensemble) of trees. Grow each tree on an independent bootstrap sample from the data.

2. At each node:
   
   - Randomly select a subset of features out of all features (independently for each node).
   - Find the best split on the selected features.
   - Grow the trees to maximum depth.

3. Prediction time
   
   - Vote the trees to get predictions for new example.

### Injecting Randomness

To ensure that the trees in the random forest are different we inject randomness in two ways:

1. Data: **Build each tree on a bootstrap sample** (i.e., a sample drawn **with replacement** from the training set)
2. Features: **At each node, select a random subset of features** (controlled by `max_features` in `scikit-learn`) and look for the best possible test involving one of these features

### Hyperparameters

- `n_estimators`: number of decision trees (higher = more complexity)
- `max_depth`: max depth of each decision tree (higher = more complexity)
- `max_features`: the number of features you get to look at each split (higher = more complexity)

#### Number of trees and fundamental trade-off

- Above: seems like we’re beating the fundamental “tradeoff” by increasing training score and not decreasing validation score much.
- This is the promise of ensembles, though it’s not guaranteed to work so nicely.

More trees are always better! We pick less trees for speed.

### Strengths and Weaknesses

Strengths:

- Usually one of the best performing off-the-shelf classifiers without heavy tuning of hyperparameters
- Don’t require scaling of data
- Less likely to overfit
- Slower than decision trees because we are fitting multiple trees but can easily parallelize training because all trees are independent of each other
- In general, able to capture a much broader picture of the data compared to a single decision tree.

Weaknesses:

- Require more memory
- Hard to interpret
- Tend not to perform well on high dimensional sparse data such as text data

Make sure to set the `random_state` for reproducibility. Changing the `random_state` can have a big impact on the model and the results due to the random nature of these models. Having more trees can get you a more robust estimate.

## Gradient Boosted Trees

Another popular and effective class of tree-based models is gradient boosted trees.

- No randomization.
- The key idea is combining many simple models called weak learners to create a strong learner.
- They combine multiple shallow (depth 1 to 5) decision trees
- They build trees in a serial manner, where each tree tries to correct the mistakes of the previous one.

### Hyperparameters

`n_estimators`

- control the number of trees to build

`learning_rate`

- controls how strongly each tree tries to correct the mistakes of the previous trees
- higher learning rate means each tree can make stronger corrections, which means more complex model

### Models

We’ll not go into the details. We’ll look at brief examples of using the following three gradient boosted tree models.

- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)
- [CatBoost](https://catboost.ai/docs/concepts/python-quickstart.html)

#### [XGBoost](https://xgboost.ai/about)

Not part of `sklearn` but has similar interface. Install it in your conda environment: `conda install -c conda-forge xgboost`.

- Supports missing values
- GPU training, networked parallel training
- Supports sparse data
- Typically better scores than random forests

#### [LightGBM](https://lightgbm.readthedocs.io/)

Not part of `sklearn` but has similar interface. Install it in your conda environment: `conda install -c conda-forge lightgbm`.

- Small model size
- Faster
- Typically better scores than random forests

#### [CatBoost](https://catboost.ai/)

Not part of `sklearn` but has similar interface. Install it in your conda environment: `conda install -c conda-forge catboost`.

- Usually better scores but slower compared to `XGBoost` and `LightGBM`

## Averaging

Earlier we looked at a bunch of classifiers. What if we use all these models and let them vote during prediction time?

This `VotingClassifier` will take a *vote* using the predictions of the constituent classifier pipelines.

Main parameter: `voting`

- `voting='hard'`
  
  - it uses the output of `predict` and actually votes.

- `voting='soft'`
  
  - with `voting='soft'` it averages the output of `predict_proba` and then thresholds / takes the larger.

- The choice depends on whether you trust `predict_proba` from your base classifiers - if so, it’s nice to access that information.

What happens when you `fit` a `VotingClassifier`?

- It will fit all constituent models.

In short, as long as the different models make different mistakes, this can work.

Why not always do this?

1. `fit`/`predict` time.
2. Reduction in interpretability.
3. Reduction in code maintainability (e.g. Netflix prize).

You can combine

- completely different estimators, or similar estimators.
- estimators trained on different samples.
- estimators with different hyperparameter values.

## Stacking

Instead of averaging the outputs of each estimator, instead use their outputs as *inputs to another model*.

By default for classification, it uses logistic regression.

- We don’t need a complex model here necessarily, more of a weighted average.
- The features going into the logistic regression are the classifier outputs, *not* the original features!
- So the number of coefficients = the number of base estimators!

It is doing cross-validation by itself by default (see [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html))

- It is fitting the base estimators on the training fold
- And the predicting on the validation fold
- And then fitting the meta-estimator on that output (on the validation fold)

Randomly generate a bunch of models with different hyperparameter configurations, and then stack all the models.

What is an advantage of ensemble models as opposed to just choosing one of them?

- You may get a better score.

What is an disadvantage of ensemble models as opposed to just choosing one of them?

- Slower, more code maintenance issues.
