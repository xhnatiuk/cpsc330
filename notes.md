Prof: vA-rA-dA (said quickly) or V or Ada

kvarada@cs.ubc.ca

# Machine Learning Fundamentals

The following code can be used to count the different values that occur in our target:`y.value_counts()`

**Baseline:** a simple machine learning algorithm based on simple rules of thumb. Provides a point of reference for comparing your machine learning model. 

* `sklearn`'s baseline model for classification is `DummyClassifier`.  The most frequent baseline will just always predict the most frequent label in the training set.

* `sklearn`'s baseline model for regression is `DummyRegressor` which predicts mean, median, or constant value of the training set for all examples. 



Steps to train a classifier using `sklearn`:

1. Read the data
   * `df = pd.read_csv("./my_data.csv")`
2. Create X (Feature vectors) and y (Target)
   * `X = df.drop(columns=["target"])`
   * `y = df["target"]`
3. Create a classifier object
   * `import` the appropriate classifier (`from sklearn.dummy import DummyClassifier`)
   * create an instance of the classifier (`dummy = DummyClassifier(strategy="most_frequent")`)
4. `fit` the classifier
   * `dummy.fit(X,y)`
5. `predict` on new examples
   * `dummy.predict(X)`
6. `score` the model
   * `dummy.score(X, y)`

In `sklearn` we can evaluate our model using the `score` function. The score function calls `predict` on `X` and compares the predictions with the targets (`y`). 

1. For classification problems, `score` gives the **accuracy** of the model:

$$
accuracy = \frac{CorrectPredictions}{TotalExamples}
$$

2. For regression problems, `score` returns the R<sup>2</sup> score. Its maximum is 1 for perfect predictions, negative means the model is performing really badly, and for the `DummyRegressor` it will be close to 0. 

Sometimes people will measure **error** which is generally `1 - accuracy`. 



## Terminology

### Types of Machine Learning

**Motivation**: We can use machine learning to save time and customize and scale products by analyzing large amounts of data and inferring patterns. 

**Supervised Machine Learning**: Training a model from input data and its corresponding targets to predict targets for unseen data.

**Unsupervised Learning**: Training a model to find patterns in a dataset, typically an unlabeled dataset. 

**Reinforcement Learning**: A family of algorithms for finding suitable actions to take in a given situation in order to maximize a reward.

**Recommendation Systems**: Predict the "rating" or "preference" a user would give to an item.  

In **supervised learning**, training data comprises a set of features (X) and their corresponding targets (y). We wish to find a **model function **(f) that relates X to y. Then use that model function to **predict the targets** of new examples. 

In **unsupervised learning** training data consists of observations (X) **without any corresponding targets**. Unsupervised learning could be used to **group similar things together in X** or to provide a **concise summary** of the data.

### Data

In supervised machine learning, the input data is typically organized into **tabular** format, where rows are **examples** (n) and columns are **features** (X, number of features denoted with d). One of the columns is typically the **target** (y). 

**Examples**: rows, samples, records, instances

**Features**: inputs, predictors, explanatory variables, regressors, independent variables, covariates

**Targets**: outputs, outcomes, response variable, dependent variable, labels (if categorical)

**Training**: learning, fitting

### Supervised Learning Problems

In supervised machine learning, there are two main types of learning problems based on what we are trying to predict:

1. **Classification**: predicting among two or more discrete classes. 
   * Ex: Predict whether a patient has a disease or not
2. **Regression**: predicting a continuous value
   * Ex: Predict housing prices



## Decision Trees

We want to predict targets using a rule-based algorithm with a number of if else statements. Decision trees are inspired by the 20-questions game. 

A decision tree is  machine learning algorithm that derives such rules from our data in a principled way. 

```python
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("./my_data.csv")
X = data.drop(columns=["target"])
y = data["target"]
X_binary = X.copy()
columns = ["grade_1", "grade_2"]

model = DecisionTreeClassifier()
model.fit(X_binary, y)
model.score(X_binary, y)
```

To visualize the tree we can use:

```python
import re 
import graphviz
from sklearn.tree import export_graphviz

def display_tree(feature_names, tree, counts=False):
    """ For binary classification only """
    dot = export_graphviz(
        tree,
        out_file=None,
        feature_names=feature_names,
        class_names=tree.classes_.astype(str),
        impurity=False,
    )    
    # adapted from https://stackoverflow.com/questions/44821349/python-graphviz-remove-legend-on-nodes-of-decisiontreeclassifier
    # dot = re.sub('(\\\\nsamples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])(\\\\nclass = [A-Za-z0-9]+)', '', dot)
    if counts: 
        dot = re.sub("(samples = [0-9]+)\\\\n", "", dot)
        dot = re.sub("value", "counts", dot)
    else:
        dot = re.sub("(\\\\nsamples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])", "", dot)
        dot = re.sub("(samples = [0-9]+)(\\\\nvalue = \[[0-9]+, [0-9]+\])\\\\n", "", dot)

    return graphviz.Source(dot)
```

* The counts tell us how many of our rows/examples fail and pass the criteria. 

At each node we determine the feature most useful for classification, and the threshold to be used. Features can be binary, categorical, or continuous. Our goal is to minimize **impurity** at each question. Common criteria for measuring/minimizing impurity: gini index (used by `skitlearn`), information gain, and cross entropy. We will not learn more about these criteria. 

### Regression

We can also use decision trees for regression problems. Instead of using gini, we use some other criteria for splitting. A common one is mean squared error (MSE). 

`scikit-learn`,  supports regression using decision trees with `DecisionTreeRegressor`

* `fit` and `score` work similarly to classification
* `score` returns R<sup>2</sup> score.
  * The maximum is 1 for perfect predictions
  * It can be negative, which is very bad. Worse than `DummyRegressor` 

### Parameters

The decision tree learns two things: 

* the best feature to split on
* the threshold for the feature to split on at each node

When you call `fit` a bunch of values get set, such as the ones listed above. These are the **parameters** of the model. They are learned by the algorithm from the data during training. We need them to make predications. 

**Hyperparameters**: the parameters you can set to control the training, set before calling `fit`. Specified based on: expert knowledge, heuristics, or systematic/automated optimization. 



When we use the default hyperparameters for the decision tree classifier, it will ensure all leaf nodes are pure. We can control this using the `max_depth` hyperparameter, which is the length of the longest path from tree root to a leaf.

* When a leaf node has no impurity, all of the data that reaches it will be labeled as one class. 
* If a leaf node is not pure, it will just assign the majority class at that particular node. 

**Decision Stump**: a decision tree with only one split (`max_depth=1`).

Some other commonly used hyperparameters of a decision tree are:

* `min_samples_split`
* `min_samples_leaf`
* `max_leaf_nodes`

There are many more and they are outlined in the documentation. 



### Decision Boundary

Another way we can think about models is to ask: what sort of test examples will the model classify as positive and which will it classify as negative?

We can visualize the decision boundary using the following:

```python
from utils import *
import matplotlib.pyplot as plt
import mglearn

# just boundary
def plot_tree_decision_boundary(
    model, X, y, x_label="x-axis", y_label="y-axis", eps=None, ax=None, title=None
):
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = "max_depth=%d" % (model.tree_.max_depth)

    mglearn.plots.plot_2d_separator(
        model, X.to_numpy(), eps=eps, fill=True, alpha=0.5, ax=ax
    )
    mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 1], y, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

# boundary and tree
def plot_tree_decision_boundary_and_tree(
    model, X, y, height=6, width=16, x_label="x-axis", y_label="y-axis", eps=None
):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(width, height),
        subplot_kw={"xticks": (), "yticks": ()},
        gridspec_kw={"width_ratios": [1.5, 2]},
    )
    plot_tree_decision_boundary(model, X, y, x_label, y_label, eps, ax=ax[0])
    ax[1].imshow(tree_image(X.columns, model))
    ax[1].set_axis_off()
    plt.show()
```

As we increase our `max_depth` our model and our decision boundary get more complicated. 

## Learning Objectives

- explain the motivation to study machine learning;
- explain supervised machine learning;



- identify whether a given problem could be solved using supervised machine learning or not;
- differentiate between supervised and unsupervised machine learning;
- explain machine learning terminology such as features, targets, predictions, training, and error;
- differentiate between classification and regression problems;
- use `DummyClassifier` and `DummyRegressor` as baselines for machine learning problems;
- explain the `fit` and `predict` paradigm and use `score` method of ML models;
- broadly describe how decision tree prediction works;
- use `DecisionTreeClassifier` and `DecisionTreeRegressor` to build decision trees using `scikit-learn`;
- visualize decision trees;
- explain the difference between parameters and hyperparameters;
- explain the concept of decision boundaries;
- explain the relation between model complexity and decision boundaries.

# 3: ML Fundamentals

In machine learning we want to glean information from labeled data so that we can label **new unlabeled** data. 

The fundamental goal of machine learning is **generalization**:

> To generalize beyond what we see in the training examples.

We only have access to a limited amount of training data and we want to learn a mapping function which will predict targets reasonably well for examples beyond this data. 

 ## Types of Errors

Given a model M, we are concerned about two different types of errors:

1. Error on the training data: *error <sub> training </sub>(M)*
2. Error on the entire distribution D of data: *error <sub> D </sub>(M)*

We are interested in the error on the entire distribution, but we do not have access to the entire distribution!

## Data Splitting

A common way to approximate generalization data is to use **data splitting**

* Keep aside some randomly selected portion from the training data
* `fit` (train) a model on the training portion only
* `score` (assess) the trained model on the set aside data, to get a sense of how well the model will be able to generalize
* Pretend that the kept aside data is representative of the real distribution D of data

We can do this using `sklearn.model_selection.train_test_split`.

* we can pass in X and y, or a dataframe with both X and y in it. 
* We can specify the train or test split sizes using the `train_size` or `test_size` parameters. Default `testsize` is 0.25.
* There is no hard and fast rule on what split sizes we should use. It mostly depends on how much data is available. Common splits are 90/10, 80/20, and 70/30.
* The `random_sate` argument controls the data shuffle before splitting. When we run this function with the same `random_state` we should get exactly the same split - which is useful if you need reproducible results. 

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)  # 80%-20% train test split on X and y

```

Usually when we do machine learning, we split the data before doing anything and put the test data in an imaginary chest lock - we do not touch it while we are training our model. 

We can use `.shape` to check the shape of our data splits. 

Sometimes we want to keep the target in the train split for exploratory data analysis (EDA) or for visualization. In this case we will create a `train_df` and a `test_df`. 

```python
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=123
)  # 80%-20% train test split on df
X_train, y_train = train_df.drop(columns=["country"]), train_df["country"]
X_test, y_test = test_df.drop(columns=["country"]), test_df["country"]
```

Now after we train our model we can examine the train and test accuracies:

```python
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
print("Train accuracy:   %0.3f" % model.score(X_train, y_train))
print("Test accuracy:   %0.3f" % model.score(X_test, y_test))
```

### Train/Validation/Test Split

Sometimes it is a good idea to have separate data for hyperparameter tuning. We can do this by creating validation data, by further splitting our train split. 

* We do NOT `fit` (train) on our validation data.
* It is only used for scoring the model; predicting how well it generalizes. 
* We use it to tune our hyperparameters, before scoring on our test data. 

The terms test data and validation data are not widely agreed upon. In this course:

* We will use validation data to refer to data where we have access to the target values - used only for tuning hyperparameters, and not for training our model. 
* We will use test data to refer to data where we have access to the target values - used only once at the end to evaluate the performance of the best performing model on our validation set. 

### Deployment Data

After we build and finalize our model we will deploy it. Our model will then have to deal with new data, in the wild. We can use the term deployment data to refer to data that we do not have access to the target values for. Deployment error is what we *truly* care about. 

* We use validation and test errors as proxies for deployment error
* If our model does well on the validation and test data, we hope that it will generalize well to the deployment data. 

| Data Type  | `fit`              | `score`            | `predict`          |
| ---------- | ------------------ | ------------------ | ------------------ |
| Train      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Validation |                    | :heavy_check_mark: | :heavy_check_mark: |
| Test       |                    | once               | once               |
|            |                    |                    | :heavy_check_mark: |

We can typically expect that
$$
E_{train} < E_{validation} < E_{test} < E_{deployment}
$$

#### Problems with Single Splits

Only using a portion of our data for training and a portion for validation. If our dataset is small we may end up with a tiny training and/or validation set. You may be unlucky with your data splitting such that they do not align well, or represent your test data. 

### Cross Validation

Cross validation provides a solution to this problem. 

1. Split the training data into k folds (k > 2, often k = 10).
2. Each fold gets a turn being the validation set.
3. Each fold gets a score, and we usually average our k results
   * Its good to examine the variation in the scores across folds
   * Gives a more robust measure of error on unseen data

Note that cross validation doesn't shuffle the data; that is done in `train_test_split`.

`cross__val_score`: gives us a list of validation scores for each fold.

- It creates `cv` folds on the data.
- In each fold, it fits the model on the training portion and scores on the validation portion.

```python
model = DecisionTreeClassifier(max_depth=4)
cv_scores = cross_val_score(model, X_train, y_train, cv=10) # array of scores
print(f"Average cross-validation score = {np.mean(cv_scores):.2f}")
print(f"Standard deviation of cross-validation score = {np.std(cv_scores):.2f}")
```

`cross_validate` is a more powerful version of `cross_val_score`. It allows us to access the training and validation scores.

```python
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
pd.DataFrame(scores) # prints results
pd.DataFrame(pd.DataFrame(scores).mean()) # prints means of results
```

## Typical Supervised Machine Learning

1. Read training data
2. Split data into train and test portions `X_train`, `y_train`, `X_test`, and `y_test`
3. Optimize hyperparameters using cross-validation on the training portion of our data
4. Asses our best performing model on the test portion of our data

What we care about is test error, which tells us how well our model can be generalized. If this error is reasonable, we will deploy our model - to be used on new unseen examples (deployment data)

## The Fundamental Trade-Off & The Golden Rule

The fundamental tradeoff of supervised learning:

> As you increase model complexity, E<sub>train</sub> tends to go down, but E<sub>valid</sub> - E<sub>train</sub> tends to go up. 

This is also known as the **bias** vs **variance** tradeoff. 

**Bias**: the tendency to consistently learn the same wrong thing (high bias corresponds to underfitting)

**Variance**: the tendency to learn random things irrespective of the real signal (high variance corresponds to overfitting)

We want to avoid both underfitting and overfitting. We want to be consistent with our training data but not rely on it too much. 

* One strategy is to minimize our cross validation error.
* My Strategy: Generally to accomplish this we want to maximize our validation score whilst minimizing the gap between our training and validation scores. 

### Types of Errors

There are 4 types of errors:

* E<sub>train</sub>: training error (mean train error from cross validation)
* E<sub>valid</sub>: validation error (mean test error from cross validation)
* E<sub>test</sub>: test error
* E<sub>best</sub>: best possible error you could get for a given problem
  * This is an imaginary error, as we cannot actually calculate this in real life because we will never have the full dataset. 

### Underfitting

If your model is too simple, like `DummyClassifier` or `DecisionTreeClassifier` with `max_depth=1`, it’s not going to pick up on some random quirks in the data but it won’t capture any useful patterns in the training data.

* The model won’t be very good in general. Both train and validation errors would be high. This is **underfitting**.
* The gap between train and validation error is going to be lower.

Generally:
$$
E_{best} < E_{train} ≲ E_{valid}
$$

### Overfitting

The problem of failing to be able to generalize to the validation data or test data is called **overfitting**.

If your model is very complex, like a `DecisionTreeClassifier(max_depth=None)`, then you will learn unreliable patterns in order to get every single training example correct.

* The training error is going to be very low but there will be a  big gap between the training error and the validation error. This is **overfitting**.

* In general, if E<sub>train</sub> is low, we are likely to be in the overfitting scenario. It is fairly common to have at least a bit of this.

- So the validation error does not necessarily decrease with the training error.

Generally:
$$
E_{train} < E_{best} < E_{valid}
$$

### The Golden Rule

Even though we care the most about the test error **the test data CANNOT influence the training phase in any way** .

* We need to be very careful not to violate it while developing a pipeline
* Even experts end up breaking this rule sometimes which leads to misleading results and a lack of generalization on deployment data. 

We can avoid violating the golden rule by separating our test data early on and never touching it until its time to score. Here is our general workflow:

1. Splitting: split the data X and y into X_train, X_test, y_train, y_test, or train_df, test_df, using `train_test_split`
2. Select the best model using cross-validation: use `cross_validate` to check validation and training scores/error
3. Scoring on Test Data: finally score on the test data with the chosen hyperparameters to examine the generalization performance.



## Learning Objectives

- explain how decision boundaries change with the `max_depth` hyperparameter;
- explain the concept of generalization;
- appropriately split a dataset into train and test sets using `train_test_split` function;
- explain the difference between train, validation, test, and “deployment” data;
- identify the difference between training error, validation error, and test error;
- explain cross-validation and use `cross_val_score` and `cross_validate` to calculate cross-validation error;
- recognize overfitting and/or underfitting by looking at train and test scores;
- explain why it is generally not possible to get a perfect test score (zero test error) on a supervised learning problem;
- describe the fundamental tradeoff between training score and the train-test gap;
- state the golden rule;
- start to build a standard recipe for supervised learning:  train/test split, hyperparameter tuning with cross-validation, test on  test set.



# 4: k-NNs and SVM with RBF Kernel

**Analogy Based Models**: An intuitive way to classify test data is by finding the most "similar" examples from the training set and using that label for the test example.

Ex: facial recognition, recommendation systems

**Feature Vector**: composed of feature values associated with an example.

A common way to calculate the distance between vectors is calculating the Euclidean distance. 

The Euclidean distance between to vectors u and v is defined as:
$$
distance(u, v) = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}
$$
`scikit-learn` supports a number of distance metrics including Euclidean distance

`from sklearn.metrics.pairwise import euclidean_distances`

## Parametric Models vs Non-parametric Models

A simple way to think about this is, do you need to store at least O(n) worth of stuff to make predictions, if so it is non-parametric:

* Ex: k-NN is a classic example of a non-parametric model
* Ex: decision stump is a parametric model

These terms are often used differently by statisticians so be careful. 

### Curse of Dimensionality

This is the second most important problem in machine learning, after over/underfitting

- Affects all learners but especially bad for nearest-neighbor.
- k-NN usually works well when the number of dimensions d is small but things fall apart quickly as d goes up.
- If there are many irrelevant attributes, k-NN is hopelessly confused because all of them contribute to finding similarity between examples.
- With enough irrelevant attributes the accidental similarity swamps out meaningful similarity and k-NN is no better than random guessing.

## k-nearest neighbors algorithm 

Given a new data point, predict the class of the point by finding the k "closest" data points in the training set.

- To understand analogy-based algorithms it’s useful to think of data as points in a high dimensional space.
- Our `X` represents the the problem in terms of relevant **features** (d) with one dimension for each **feature** (column).
- Examples are **points in a** **d-dimensional space**.

In machine learning we often deal with high dimensional problems where the examples are hard to visualize:

* d ~ 20 is considered low dimensional
* d ~ 1000 is considered medium dimensional
* d ~ 100,000 is considered high dimensional

```python
from sklearn.neighbors import KNeighborsClassifier

k=3
knn = KNeighborsClassifier(n_neighbors=k)
scores = cross_validate(knn, X_train, y_train, return_train_score=True)
print("Mean validation score %0.3f" % (np.mean(scores["test_score"])))
```

Our hyperparameter here is `n_neighbors` defining how many neighbors should vote during predictions. 

* as k increases we underfit our model; low values of k overfit our model

Other useful hyperparameters include `weights` which allows you to assign a higher weight to the examples which are closer to the vector in question. 

### Pros

* Easy to understand and interpret
* Simple hyperparameter k controlling the fundamental tradeoff
* Can learn very complex functions if given enough data
* Lazy learning: takes no time to `fit` (just stores all the examples)

### Cons

- Can be potentially be VERY slow during prediction time, especially when the training set is very large. Has to calculate distances to every single example.
- Often not that great test accuracy compared to the modern approaches.
- It does not work well on datasets with many features or where most feature values are 0 most of the time (sparse datasets).

### Regression

* In k-NN regression we take the average of the k-nearest neighbors
* We can also have weighted regression

## Support Vector Machines (SVMs) with RBF kernel

We will do a super high-level overview. Our goals here are:

* Use `scikit-learn`'s SVM model
* Broadly explain the notion of support vectors
* Broadly explain the similarities and differences between k-NNs and SVM RBFs
* Explain how `C` and `gamma` hyperparameters control the fundamental tradeoff.

Another popular similarity based algorithm is SVM RBFs.  Superficially, SVM RBFs are more like weighted k-NNs. RBF stands for radial basis functions. We will not learn about this.

* The decision boundary is defined by **a set of positive and negative examples** and **their weights** together with **their similarity measure**.
* A test example is labeled positive if on average it looks more like positive examples than the negative examples.

The primary difference between k-NNs and SVM RBFs is that

- Unlike k-NNs, SVM RBFs only remember the key examples (support vectors). So it’s more efficient than k-NN.

* SVMs use a different similarity metric which is called a “kernel” in SVM land. A popular kernel is Radial Basis Functions (RBFs)

* They usually perform better than k-NNs!

```python
from sklearn.svm import SVC

svm = SVC(gamma=0.01)
scores = cross_validate(svm, X_train, y_train, return_train_score=True)
print("Mean validation score %0.3f" % (np.mean(scores["test_score"])))
```

We can think of SVM with RBF kernel as KNN with a smooth decision boundary.

- Each training example either is or isn’t a “support vector”.
  - This gets decided during `fit`.
- **Main insight**: the decision boundary only depends on the support vectors.

* We can examine the support vectors for the model using `svm.support`

### Hyperparameters

`gamma` controls the complexity (fundamental trade-off), just like other hyperparameters we’ve seen.

- larger `gamma` means more complex

* smaller `gamma` means less complex

`C` *also* affects the fundamental tradeoff

- larger `C` means more complex

* smaller `C` means less complex

Because we have more than one hyperparameter we need new methods to test combinations of hyperparameters to find our optimized model. 

### Regression

SVMs can also be used for regression problems `sklearn.svm.SVR`

## Learning Objectives

- explain the notion of similarity-based algorithms;

- broadly describe how

- -NNs use distances;

- discuss the effect of using a small/large value of the hyperparameter
  when using the

- -NN algorithm;

- describe the problem of curse of dimensionality;

- explain the general idea of SVMs with RBF kernel;

- broadly describe the relation of gamma and C hyperparameters of SVMs with the fundamental tradeoff.

# 5: Preprocessing & Pipelines

Very often real-world datasets need preprocessing before we can use them to build machine learning models. 

When trying to use KNNs or SVM RBFs with real data, our distance will be completely dominated by the features with the largest values. Basically we are ignoring the features with small values! This is a problem because features on small scales can still be highly informative - there is no reason to ignore them. We do not want our models to be sensitive to scale. 

We will use `scikit-learn`'s `StandardScaler` which is a `transformer`. 

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # create feature trasformer object
scaler.fit(X_train)  # fitting the transformer on the train split
X_train_scaled = scaler.transform(X_train)  # transforming the train split
X_test_scaled = scaler.transform(X_test)  # transforming the test split
pd.DataFrame(X_train_scaled, columns=X_train.columns).head()
```

`sklearn` uses `fit` and `transform` paradigms for feature transformations.

- We `fit` the transformer on the train split and then transform the train split as well as the test split.
- We apply the same transformations on the test split, but we never fit on it!

**Estimators**:

Suppose `model` is a classification or regression model.

```python
model.fit(X_train, y_train)
X_train_predictions = model.predict(X_train)
X_test_predictions = model.predict(X_test)
```

**Transformers**:

Suppose `transformer` is a transformer used to change the input representation, for example, to tackle missing values or to scales numeric features.

```python
transformer.fit(X_train, [y_train]) # y_train is optional, we dont really need it
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

- You can pass `y_train` in `fit` but it’s usually ignored. It allows you to pass it just to be consistent with usual usage of `sklearn`’s `fit` method.
- You can also carry out fitting and transforming in one call using `fit_transform`. But be mindful to use it only on the train split and **not** on the test split.

### Exploratory Data Analysis

`df.head()`: shows us some sample rows of our dataframe.

`df.info()`: gives us the column number, name, count, and datatype for our features. Note that strings will be datatype object.

`df.describe()`: gives us the count, mean, std, min, max, and more information about each feature. 

## Common Preprocessing Techniques

Some commonly performed feature transformation include:

- **Imputation**: Tackling missing values
- **Scaling**: Scaling of numeric features
- **One-hot encoding**: Tackling categorical variables

Numeric Columns:

* imputation
* scaling

Categorical Columns:

* imputation
* one-hot encoding

### Adding New Features

We can add new calculated features to our dataset, so long as they do not use any global information in the data - they must only look at other data in their row. Otherwise we are breaking the golden rule. If we are using features to calculate new features we may want to consider removing the old features. 

### Imputation

We want to be able to replacing missing values with some reasonable values.

* For categorical values we may want to use the most frequent value
* In numeric columns we may want to use the mean or median of the column

`SimpleImputer` is a transformer in `sklearn` for dealing with this problem.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

* Note that `imputer.transform` returns an `numpy` array and not a dataframe

### Scaling

- This problem affects a large number of ML methods.
- A number of approaches to this problem. We are going to look into two most popular ones.

| Approach        | What it does                     | How to update X                                        | sklearn implementation                                       |
| --------------- | -------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| normalization   | sets range to [0, 1]             | `X-= np.min(X, axis=0)`<br />`X /= np.max(X, axis=0)`  | [`MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) |
| standardization | sets sample mean to 0, s.d. to 1 | `X-= np.mean(X, axis=0)`<br />`X /= np.std(X, axis=0)` | [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) |

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() # or = MinMaxScaler
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
pd.DataFrame(X_train_scaled, columns=X_train.columns)

```

* `fit_transform` combines the fit and the transform step. We can use this safely on our train data but DO NOT use it on your test data. 
* Big difference in the KNN training performance after scaling the data.
* Not a big difference between KNN using normalization vs using standardization

### Cross Validation

When we use transformers we should only be fitting on our training data NOT our test data. In the case of cross validation we are treating our validation data for each fold, like it is our test data. Therefore if we do our preprocessing before calling cross validate we will be allowing information from our validation set to leak into the training step. 

* We need to apply the same preprocessing steps to train / validation data.
* With many different transformations and cross validations the code can get messy and it is easy to leak information.

We can use pipelines to solve this issue:

```python
scores = cross_validate(pipe, X_train, y_train, return_train_score=True)
```

Using a `Pipeline` takes care of applying the `fit_transform` on the train portion and only `transform` on the validation portion in each fold.

### Pipelines

`scikit-learn Pipeline` allows us to define a pipeline of transformers with a final estimator. Let’s combine the preprocessing and model with pipeline:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", KNeighborsRegressor()),
    ]
)
```

- Syntax: pass in a list of steps.
- The last step should be a **model/classifier/regressor**.
- All the earlier steps should be **transformers**.

#### `make_pipeline`

- Shorthand for `Pipeline` constructor
- Does not permit naming steps
- Instead the names of steps are set to lowercase of their types automatically; `StandardScaler()` would be named as `standardscaler`

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor()
)
pipe.fit(X_train, y_train)
```

- Note that we are passing `X_train` and **not** the imputed or scaled data here.

When you call `fit` on the pipeline, it carries out the following steps:

- Fit `SimpleImputer` on `X_train`
- Transform `X_train` using the fit `SimpleImputer` to create `X_train_imp`
- Fit `StandardScaler` on `X_train_imp`
- Transform `X_train_imp` using the fit `StandardScaler` to create `X_train_imp_scaled`
- Fit the model (`KNeighborsRegressor` in our case) on `X_train_imp_scaled`

Note that we are passing original data to `predict` as well. This time the pipeline is carrying out following steps:

* Transform `X_train` using the fit `SimpleImputer` to create `X_train_imp`
* Transform `X_train_imp` using the fit `StandardScaler` to create `X_train_imp_scaled`
* Predict using the fit model (`KNeighborsRegressor` in our case) on `X_train_imp_scaled`.

## Categorical Features

In `scikit-learn`, most algorithms require numeric inputs. For example, we cannot build a KNN model on categorical features since they are non-numeric, and therefore we have no way to calculate distances. 

Decision trees could theoretically work with categorical features, however, the `sklearn` implementation does not support this.

We can transform categorical features to numeric ones so that we can use them in the model.

* [Ordinal encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html) (occasionally recommended)
* One-hot encoding (recommended in most cases)

### Ordinal Encoding 

Here we simply assign an integer to each of our unique categorical labels. We can use sklearn’s [`OrdinalEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html).

```
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
enc.fit(X)
X_ord = enc.transform(X)
df = pd.DataFrame(
    data=X_ord,
    columns=["language_enc"],
    index=X.index,
)
pd.concat([X, df], axis=1)
```

What’s the problem with this approach?

- We have imposed ordinality on the categorical data.
- For example, imagine when you are calculating distances. Is it  fair to say that French and Hindi are closer than French and Spanish?
- In general, label encoding is useful if there is ordinality in your data and capturing it is important for your problem, e.g., `[cold, warm, hot]`.

What’s the problem here? The encoder doesn’t know the order.

* We can examine unique categories manually, order them based on our  intuitions, and then provide this human knowledge to the transformer.
* `X["ordinal_feature"].unique()` will give us all of the unique categories for our categorical data
* We can then make an ordered list, note that if you use the reverse order of the categories, it wouldn’t matter.
  * `class_attendance_levels = ["Poor", "Average", "Good", "Excellent"]`
* Ensure that we have all the same categories in our manual ordering
  * `assert set(class_attendance_levels) == set(X_toy["class_attendance"].unique())`
* Then we add them to the transformer

```python
oe = OrdinalEncoder(categories=[class_attendance_levels], dtype=int)
oe.fit(X[["class_attendance"]])
ca_transformed = oe.transform(X[["class_attendance"]])
df = pd.DataFrame(
    data=ca_transformed, columns=["class_attendance_enc"], index=X.index
)
pd.concat([X, df], axis=1).head(10)
```

#### More than one Ordinal Column

We can pass the manually ordered categories when we create an `OrdinalEncoder` object as a list of lists. If you have more than one ordinal columns

- manually create a list of ordered categories for each column
- pass a list of lists to `OrdinalEncoder`, where each inner list corresponds to manually created list of ordered categories for a corresponding ordinal column.

### One-hot Encoding (OHE)

Create new binary columns to represent our categories.

* If we have c categories in our column.
  * We create c new binary columns to represent those categories.

* Example: Imagine a language column which has the information on whether you

* We can use sklearn’s [`OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) to do so.

One-hot encoding is called one-hot because only one of the newly created features is 1 for each data point.

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
enc.fit(X)
X_ohe = enc.transform(X)
pd.DataFrame(
    data=X_ohe,
    columns=enc.get_feature_names(["language"]),
    index=X.index,
)
```

We can look at the new features created using the `categories_` attribute, ex `enc.categories_`

- By default, `OneHotEncoder` also creates sparse features.
- You could set `sparse=False` to get a regular `numpy` array.
- If there are a huge number of categories, it may be beneficial to keep them sparse.
- For smaller number of categories, it doesn’t matter much.

#### Binary Features

Sometimes you have features with only two possible categories. 

* If we apply `OheHotEncoder` on such columns, it’ll create two columns, which seems wasteful, as we  could represent all information in the column in just one column with  say 0’s and 1’s with presence of absence of one of one of the  categories.
* You can pass `drop="if_binary"` argument to `OneHotEncoder` in order to create only one column in such scenario.

#### Many Categories

Do we have enough data for rare categories to learn anything meaningful?

- How about grouping them into bigger categories?
  - Example: country names into continents such as “South America” or “Asia”
- Or having “other” category for rare cases?

### Picking Features

- Do you want to use certain features such as **gender** or **race** in prediction?
- Remember that the systems you build are going to be used in some applications.
- It’s extremely important to be mindful of the consequences of including certain features in your predictive model.

## Learning Objectives

- explain motivation for preprocessing in supervised machine learning;
- identify when to implement feature transformations such as  imputation, scaling, and one-hot encoding in a machine learning model  development pipeline;
- use `sklearn` transformers for applying feature transformations on your dataset;
- discuss golden rule in the context of feature transformations;
- use `sklearn.pipeline.Pipeline` and `sklearn.pipeline.make_pipeline` to build a preliminary machine learning pipeline.

# 6: Preprocessing Text & Transformers

## sklearn’s [`ColumnTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)

- In most applications, some features are categorical, some are continuous, some are binary, and some are ordinal.
- When we want to develop supervised machine learning pipelines on  real-world datasets, very often we want to apply different  transformation on different columns.
- Enter `sklearn`’s `ColumnTransformer`

Steps:

1. Identify all the different types of data we want to perform transformations on
   * numeric, categorical, ordinal, binary/passthrough (don't do any transformations), drop (features to drop)
2. Identify the transformations we want to apply
3. Create a column transformer

Each transformation is specified by a name, a transformer object, and the columns this transformer should be applied to.

```python
from sklearn.compose import ColumnTransformer
numeric_feats = ["university_years", "lab1", "lab3", "lab4", "quiz1"]  # apply scaling
categorical_feats = ["major"]  # apply one-hot encoding
ct = ColumnTransformer(
    [
        ("scaling", StandardScaler(), numeric_feats),
        ("onehot", OneHotEncoder(sparse=False), categorical_feats),
    ]
)
```

### `make_column_transformer`

- Similar to `make_pipeline` syntax, there is convenient `make_column_transformer` syntax.
- The syntax automatically names each step based on its class.

```python
from sklearn.compose import make_column_transformer

ct = make_column_transformer(
    (StandardScaler(), numeric_feats),  # scaling on numeric features
    (OneHotEncoder(), categorical_feats),  # OHE on categorical features
    ("passthrough", passthrough_feats),  # no transformations on the binary features
    ("drop", drop_feats),  # drop the drop features
)
transformed = ct.fit_transform(X)
```

- note that we can see what transformers are being applied using the `named_transformers_` attribute

When we `fit_transform`, each transformer is applied to the specified columns and the result of the transformations are concatenated horizontally.

- A big advantage here is that we build all our transformations  together into one object, and that way we’re sure we do the same  operations to all splits of the data.
- Otherwise we might, for example, do the OHE on both train and test but forget to scale the test data.

Note that the returned object is not a dataframe. So there are no column names. How can we view our transformed data as a dataframe?

* We are adding more columns. So the original columns won’t directly map to the transformed data. Let’s create column names for the transformed data.

```python
column_names = (
    numeric_feats
    + ct.named_transformers_["onehotencoder"].get_feature_names().tolist()
    + passthrough_feats
)
pd.DataFrame(transformed, columns=column_names)
```

* Note that the order of the columns in the transformed data depends upon the order of the features we pass to the `ColumnTransformer` and can be different than the order of the features in the original dataframe.

We can now pass the `ColumnTransformer` object as a step in a pipeline.

```python
pipe = make_pipeline(ct, SVC())
pipe.fit(X, y)
pipe.predict(X)
```

To apply more than one transformations we can define a pipeline inside a column transformer to chain different transformations. For example if we wanted to impute and then scale our numeric features we would do the following:

```python
ct = make_column_transformer(
    (
        make_pipeline(SimpleImputer(), StandardScaler()),
        numeric_feats,
    ),  # scaling on numeric features
    (OneHotEncoder(), categorical_feats),  # OHE on categorical features
    ("passthrough", passthrough_feats),  # no transformations on the binary features
    ("drop", drop_feats),  # drop the drop features
)
```

With multiple transformations in a column transformer, it can get tricky to keep track of everything happening inside it. We can use `set_config` to display a diagram of this.

```python
from sklearn import set_config
set_config(display="diagram")
```

![image-20211025210527787](C:\Users\Max\AppData\Roaming\Typora\typora-user-images\image-20211025210527787.png)

## Dealing with Unknown Categories

Sometimes if our dataset is small we can have data put into our validation split that then would have its own columns made by the transformers and are therefore not recognized during cross validation. 

* By default, `OneHotEncoder` throws an error because you might want to know about this.
* Simple fix:
  * Pass `handle_unknown="ignore"` argument to `OneHotEncoder`
  * It creates a row with all zeros.
  * With this approach, all unknown categories will be represented with all zeros and cross-validation will run OK now.

```python
ct = make_column_transformer(
    (
        make_pipeline(SimpleImputer(), StandardScaler()),
        numeric_feats,
    ),  # scaling on numeric features
    (
        OneHotEncoder(handle_unknown="ignore"),
        categorical_feats,
    ),  # OHE on categorical features
    (
        OrdinalEncoder(categories=[class_attendance_levels], dtype=int),
        ordinal_feats,
    ),  # Ordinal encoding on ordinal features
    (
        OneHotEncoder(drop="if_binary", dtype=int),
        binary_feats,
    ),  # OHE on categorical features
    ("passthrough", passthrough_feats),  # no transformations on the binary features
)
pipe = make_pipeline(ct, SVC())
scores = cross_validate(pipe, X, y, cv=5, return_train_score=True)
pd.DataFrame(scores)
```

### Breaking the Golden Rule

If it’s some fix number of categories. For example, if it’s something  like provinces in Canada or majors taught at UBC. We know the categories in advance and this is one of the cases where it might be OK to violate the golden rule and get a list of all possible values for the  categorical variable.

## Encoding Text Data

How can we encode text data so that we can pass it into machine learning algorithms? Some popular representations of raw text include:

- **Bag of words**
- TF-IDF
- Embedding representations

### Bag of Words (BOW)

One of the most popular representation of raw text. Ignores the syntax and word order. Has two components:

1. The vocabulary (all unique words in all documents)
2. A value indicating either the presence or absence or the count of each word in the document.

In the Natural Language Processing (NLP) community text data  is referred to as a **corpus** (plural: corpora). 

`CountVectorizer`

- Converts a collection of text documents to a matrix of word counts.
- Each row represents a “document” (e.g., a text message in our example).
- Each column represents a word in the vocabulary (the set of unique words) in the training data.
- Each cell represents how often the word occurs in the document.

Of course this is not a great representation of language, but it works surprisingly well for many tasks.

```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X_counts = vec.fit_transform(df["text_feature"])
bow_df = pd.DataFrame(
    X_counts.toarray(), columns=vec.get_feature_names(), index=df["text_feature"]
)
```

Note that `CountVectorizer` is carrying out some preprocessing such as because of the default argument values

- Converting words to lowercase (`lowercase=True`)
- getting rid of punctuation and special characters (`token_pattern ='(?u)\\b\\w\\w+\\b'`)

Note that unlike other transformers we are passing a `Series` object to `fit_transform`. For other transformers, you can define one transformer for more than one columns. But with `CountVectorizer` you need to define separate `CountVectorizer` transformers for each text column, if you have more than one text columns.

`X_counts` is of type sparse matrix. This is because:

* Most words do not appear in a given document.
* We get massive computational savings if we only store the nonzero elements.
* There is a bit of overhead, because we also need to store the locations:
  - e.g. “location (3,27): 1”.
* However, if the fraction of nonzero is small, this is a huge win.
* Code to check: 

```python
print("The total number of elements: ", np.prod(X_counts.shape))
print("The number of non-zero elements: ", X_counts.nnz)
print(
    "Proportion of non-zero elements: %0.4f" % (X_counts.nnz / np.prod(X_counts.shape))
)
```

#### `CountVectorizer` Hyperparameters

- `binary`: whether to use absence/presence feature values or counts
- `max_features`: only consider top `max_features` ordered by frequency in the corpus
- `max_df`: ignore features which occur in more than `max_df` documents
- `min_df`: ignore features which occur in less than `min_df` documents
- `ngram_range`: consider word sequences in the given range

## Learning Objectives

- use `ColumnTransformer` to build all our transformations together into one object and use it with `sklearn` pipelines;
- define `ColumnTransformer` where transformers contain more than one steps;
- explain `handle_unknown="ignore"` hyperparameter of `scikit-learn`’s `OneHotEncoder`;
- explain `drop="if_binary"` argument of `OneHotEncoder`;
- identify when it’s appropriate to apply ordinal encoding vs one-hot encoding;
- explain strategies to deal with categorical variables with too many categories;
- explain why text data needs a different treatment than categorical variables;
- use `scikit-learn`’s `CountVectorizer` to encode text data;
- explain different hyperparameters of `CountVectorizer`.

# 7: Linear Models



## Learning Objectives

- Explain the general intuition behind linear models;
- Explain how `predict` works for linear regression;
- Use `scikit-learn`’s `Ridge` model;
- Demonstrate how the `alpha` hyperparameter of `Ridge` is related to the fundamental tradeoff;
- Explain the difference between linear regression and logistic regression;
- Use `scikit-learn`’s `LogisticRegression` model and `predict_proba` to get probability scores
- Explain the advantages of getting probability scores instead of hard predictions during classification;
- Broadly describe linear SVMs
- Explain how can you interpret model predictions using coefficients learned by a linear model;
- Explain the advantages and limitations of linear classifiers
- Carry out multi-class classification using OVR and OVO strategies.

# 8: Hyperparameter Optimization & Overfitting



## Learning Objectives

- explain the need for hyperparameter optimization
- carry out hyperparameter optimization using `sklearn`’s `GridSearchCV` and `RandomizedSearchCV`
- explain different hyperparameters of `GridSearchCV`
- explain the importance of selecting a good range for the values.
- explain optimization bias
- identify and reason when to trust and not trust reported accuracies

# 9: Evaluation Metrics for Classification



## Learning Objectives

- Explain why accuracy is not always the best metric in ML.
- Explain components of a confusion matrix.
- Define precision, recall, and f1-score and use them to evaluate different classifiers.
- Broadly explain macro-average, weighted average.
- Interpret and use precision-recall curves.
- Explain average precision score.
- Interpret and use ROC curves and ROC AUC using `scikit-learn`.
- Identify whether there is class imbalance and whether you need to deal with it.
- Explain and use `class_weight` to deal with data imbalance.

# 10: Regression Metrics



## Learning Objectives

- Carry out feature transformations on somewhat complicated dataset.
- Visualize transformed features as a dataframe.
- Use `Ridge` and `RidgeCV`.
- Explain how `alpha` hyperparameter of `Ridge` relates to the fundamental tradeoff.
- Examine coefficients of transformed features.
- Appropriately select a scoring metric given a regression problem.
- Interpret and communicate the meanings of different scoring metrics on regression problems.
  - MSE, RMSE, R^2, MAPE
- Apply log-transform on the target values in a regression problem with `TransformedTargetRegressor`.

# 11: Ensembles



## Learning Objectives

- Use `scikit-learn`’s `RandomForestClassifier` and explain its main hyperparameters.
- Explain randomness in random forest algorithm.
- Use other tree-based models such as as `XGBoost` and `LGBM`.
- Employ ensemble classifier approaches, in particular model averaging and stacking.
- Explain voting and stacking and the differences between them.
- Use `scikit-learn` implementations of these ensemble methods.

# 12: Features Importance & Feature Engineering 

## Feature Correlations

One way for determining the importance of features is to look at the correlations between features and other features in our data.

* Positive Correlation: Y goes up when X goes up
* Negative Correlation: Y goes down when X goes up
* Uncorrelated: Y doesn't change when X goes up

This approach is extremely simplistic.

* It only looks at each feature in isolation
* It only looks at linear associations

Sometimes a feature only becomes important if another feature is *added* or *removed*.

### Ordinal Features

Ordinal features are the easiest to interpret. In a linear regression, if we increase our ordinal feature by 1 category, it effects our model by 1 times its learned coefficient. 

### Categorical Features

With categorical features we consider one of the categories for a feature to be the reference category. We then calculate the difference between the other categories for the feature and the reference category to interpret their effect on the model.

* Ex: if feature 1 changed from category A (reference) to category B (non-reference) if would effect our model by its difference. 
* Do we really believe these interpretations?
  * This is how predictions are being made, so yes
  * But this is likely not how the world works, so no

### Numeric Features

This is trickier than you would expect since we have scaled our numeric features. Our intuition should be that if we increase a numeric feature by 1 scaled unit then if effects our model by 1 times its learned coefficient. 

To interpret a feature we should divide the learned coefficient from the model by the scale coefficient. 

## Interpretability

The ability to interpret our models is crucial in many applications such as: banking, healthcare, and criminal justice. It can be leveraged by domain experts to diagnose systematic errors and underlying biases of complex ML systems.

In this course our definition of model interpretability is **feature importance**. There are more factors in interpretability but this is a good start. 

Feature importance does not have a sign!

* Only tells us about importance, nothing about up or down.



### Shapley Additive Explanations (SHAP)

A sophisticated measure of the contribution of each feature. We will not go into details of how it works, but we will learn how to use it. 

We can use SHAP to explain predictions on our deployment data. 

We can use average SHAP values to determine global feature importance.

Smaller SHAP values mean that we are less likely to get placed in the target class we are looking at. Synonymous with correlation? 

## Learning Objectives

* Interpret the coefficients of linear regression for ordinal, one-hot encoded categorical, and scaled numeric features.
* Explain why interpretability is important in ML.
* Use `feature_importances_` attribute of `sklearn` models and interpret its output.
* Use `eli5` to get feature importance of non `sklearn` models and interpret its output.
* Apply SHAP to assess feature importance and interpret model predictions.
* Explain force plot, summary plot, and dependence plot produced with shapely values.
