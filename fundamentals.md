**Motivation**: We can use machine learning to save time, customize, and scale products by analyzing large amounts of data and inferring patterns.

The fundamental goal of machine learning is **generalization**:

> To generalize beyond what we see in the training examples.

We only have access to a limited amount of training data and we want to learn a mapping function which will predict targets reasonably well for examples beyond this data.

# Types of Machine Learning

**Supervised Machine Learning**: Training a model from input data and its corresponding targets to predict targets for unseen data.

* Training data comprises a set of features (X) and their corresponding targets (y). 

* We wish to find a model function (f) that relates X to y. Then use that model function to predict the targets of new examples.

**Unsupervised Learning**: Training a model to find patterns in a dataset, typically an unlabeled dataset.

* Training data consists of observations (X) **without any corresponding targets**. 

* Used to **group similar things together in X** or to provide a **concise summary** of the data.

**Reinforcement Learning**: A family of algorithms for finding suitable actions to take in a given situation in order to maximize a reward.

**Recommendation Systems**: Predict the "rating" or "preference" a user would give to an item.

## Supervised Machine Learning

In supervised machine learning, there are two main types of learning problems based on what we are trying to predict:

1. **Classification**: predicting among two or more discrete classes.
   - Ex: Predict whether a patient has a disease or not
2. **Regression**: predicting a continuous value
   - Ex: Predict housing prices

Typical process:

1. Read training data
2. Split data into train and test portions `X_train`, `y_train`, `X_test`, and `y_test`
3. Optimize hyperparameters using cross-validation on the training portion of our data
4. Asses our best performing model on the test portion of our data

# The Golden Rule

Even though we care the most about the test error **the test data CANNOT influence the training phase in any way** .

- We need to be very careful not to violate it while developing a pipeline
- Even experts end up breaking this rule sometimes which leads to misleading results and a lack of generalization on deployment data.

We can avoid violating the golden rule by separating our test data early on and never touching it until its time to score. Here is our general workflow:

1. Splitting the test data
2. Select the best model using cross-validation
3. Score on the test data to determine generalization performance

# Data

In supervised machine learning, the input data is typically organized into **tabular** format, where rows are **examples** (n) and columns are **features**. One of the columns is typically the **target**.

**Features**: relevant characteristics of the problem, usually suggested by experts.  

* Tyically denoted by X

* Number of features denoted by d

* Synonyms: inputs, predictors, explanatory variables, regressors, independent variables, covariates

**Targets**: the feature we want to predict

* Tyically denoted by y

* Synonyms: outputs, outcomes, response variable, dependent variable, labels (if categorical)

**Examples**: A row of feature values.

* May or may not include the target corresponding to the feature values

* Synonyms: rows, samples, records, instances

**Training**: The process of learning the mapping between the features and the target.

* Synonyms: learning, fitting

## Data Splitting

A common way to approximate generalization error is to use **data splitting**. We generally split our data before doing anything. 

The main idea is to keep aside a randomly selected portion from the training data. We then train our model only on one portion of the data and score on the other. This should give us a good idea of how well our model will generalize. 

There is no hard and fast rule on what split sizes we should use. It mostly depends on how much data is available. Common splits are 90/10, 80/20, and 70/30.

### Train/Validation/Test Split

Sometimes it is a good idea to have separate data for hyperparameter tuning. We can do this by creating validation data, by further splitting our train split.

- We do NOT `fit` (train) on our validation data.
- It is only used for scoring the model; predicting how well it generalizes.
- We use it to tune our hyperparameters, before scoring on our test data.

The terms test data and validation data are not widely agreed upon. In this course:

- We will use validation data to refer to data used only for tuning hyperparameters, and not for training our model.
- We will use test data to refer to data used only once at the end to evaluate the performance of the best performing model on our validation set.

### Deployment Data

After we build and finalize our model we will deploy it. Our model will then have to deal with new data, in the wild. We can use the term deployment data to refer to data that we do not have access to the target values for. Deployment error is what we *truly* care about.

- We use validation and test errors as proxies for deployment error
- If our model does well on the validation and test data, we hope that it will generalize well to the deployment data.

| Data Type  | `fit`              | `score`            | `predict`          |
| ---------- | ------------------ | ------------------ | ------------------ |
| Train      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Validation |                    | :heavy_check_mark: | :heavy_check_mark: |
| Test       |                    | once               | once               |
| Deployment |                    |                    | :heavy_check_mark: |

We can typically expect that

$E_{train} < E_{validation} < E_{test} < E_{deployment}$

### Cross Validation

If our dataset is small we may end up with a tiny training and/or validation set. You may be unlucky with your data splitting such that they do not align well, or represent your test data.

Cross validation provides a solution to this problem.

1. Split the training data into k folds (k > 2, often k = 10).
2. Each fold gets a turn being the validation set.
3. Each fold gets a score, and we usually average our k results
   - Its good to examine the variation in the scores across folds
   - Gives a more robust measure of error on unseen data

# Tuning and Training

## Fitting, Parameters, and Hyperparameters

When we cann `fit` a bunch of values are 'learned' by our model and set. These are our paratmers. Before we call `fit` we can change the way that our model learns the `parameters` these are called `hyperparameters`.

**Parameters**: values learned by the model from the data during training. Used during `predict`.

**Hyperparameters**: values used to control how our model learns `parameters`. Used during `fit`.

* based on expert knowledge, heuristics, or systematic/automated optimization

## The Fundamental Tradeoff

The fundamental tradeoff of supervised learning:

> As you increase model complexity, E_{train} tends to go down, but E_{validation} - E_{train} tends to go up.

This is also known as the **bias** vs **variance** tradeoff.

**Bias**: the tendency to consistently learn the same wrong thing (high bias corresponds to underfitting)

**Variance**: the tendency to learn random things irrespective of the real signal (high variance corresponds to overfitting)

We want to avoid both underfitting and overfitting. We want to be consistent with our training data but not rely on it too much.

- One strategy: minimize our cross validation error.
- Our strategy: maximize our validation score whilst minimizing the gap between our training and validation scores.

### Underfitting

If your model is too simple, like a baseline or decision stump, it’s not going to pick up on any random quirks in the data but it also won’t capture any useful patterns in the training data.

* How to tell: training and validation error are both high

Generally: $E_{best} < E_{train} ≲ E_{valid}$

### Overfitting

If your model is very complex, like a `DecisionTreeClassifier(max_depth=None)`, you will learn unreliable patterns in order to get every single training example correct.

- How to tell: training error is low, and there is a large gap between training error and validation error

Generally: $E_{train} < E_{best} < E_{valid}$

## Parametric Models vs Non-parametric Models

Non-parametric models: need to store O(n) worth of stuff to make predicitons. 

- Ex: k-NN is a classic example of a non-parametric model
- Ex: decision stump is a parametric model

These terms are often used differently by statisticians so be careful.

### Dimensionality

**Analogy Based Model**: An intuitive way to classify test data is by finding the most "similar" examples from the training set and using that label for the test example.

Sometimes its useful to think of data as points in a high dimensional space, particularly with analogy based models.

- Our `X` represents the the problem in terms of relevant **features** (d) with one dimension for each **feature** (column).
- Examples are **points in a** **d-dimensional space**.

In machine learning we often deal with high dimensional problems where the examples are hard to visualize:

- d ~ 20 is considered low dimensional
- d ~ 1000 is considered medium dimensional
- d ~ 100,000 is considered high dimensional

#### Curse of Dimensionality

The curse of dimensionality is when there are enough irrelevant attributes that the accidental similarity between examples dominates meaningful similarity and causes the model to perform poorly. 

This is the second most important problem in machine learning, after the fundamental tradeoff. 

Affects all models but especially bad for kNNs. When the number of dimensions is small KNNs work well, but wheen there are many irrelevant attributes, kNN is hoplessly confused because they all contribute to similarity equally. With enough dimensions KNN is no better than random guessing. 

# Predicting and Scoring

In `sklearn` we can evaluate our model using the `score` function. The score function calls `predict` on `X` and compares the predictions with the targets (`y`).

1. For classification problems, `score` gives the **accuracy** of the model:

$$
accuracy = \frac{CorrectPredictions}{TotalExamples}
$$

2. For regression problems, `score` returns the R2 score. Its maximum is 1 for perfect predictions, negative means the model is performing really badly, and for the `DummyRegressor` it will be close to 0.

Sometimes people will measure **error** which is generally `1 - accuracy`.

### Decision Boundary

We use classification models to predict the class of an unseen example. Another way we can think about our models is to ask:

> What sort of test examples will the model classifiy as positive and what will it classify as negative?

The boundary between vectors that are classified as positive vs negative, is called the **decision boundary**. As models become more complicated, so does its decision boundary. 

## Types of Errors

Given a model M, we are concerned about two different types of errors:

1. Error on the training data: *error training (M)*
2. Error on the entire distribution D of data: *error D (M)*

We are interested in the error on the entire distribution, but we do not have access to the entire distribution!

There are 4 types of errors:

- $E_{train}$: training error (mean train error from cross validation)
- $E_{valid}$: validation error (mean test error from cross validation)
- $E_{test}$: test error
- $E_{best}$: best possible error you could get for a given problem
  - This is an imaginary error, as we cannot actually calculate this in real life because we will never have the full dataset.

# Preprocessing

Very often real-world datasets need preprocessing before we can use them to build machine learning models.

Some commonly performed feature transformation include:

- **Imputation**: Dealing with missing values
- **Scaling**: Scaling of numeric features
- **One-hot encoding**: Dealing with categorical variables

By Feature Type:

* Numeric: Imputation, Scaling

* Categorical: Imputation, One-hot encoding

## Transformers

When we use transformers we should only be fitting on our training data NOT our test data.

- We `fit` the transformer on the train split and then transform the train split as well as the test split.
- We apply the same transformations on the test split, but we never fit on it!

Suppose `transformer` is a transformer used to change the input representation, for example, to tackle missing values or to scale numeric features.

```python
transformer.fit(X_train, [y_train]) # y_train is optional, we dont really need it
X_train_transformed = transformer.transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

- You can pass `y_train` in `fit` but it’s usually ignored. 
- You can also carry out fitting and transforming in one call using `fit_transform`. But be mindful to use it only on the train split and **not** on the test split.

## Cross Validation and Pipelines

When we cross validate we treat our validation data in each fold like it is our test data. Therefore if we do any transformations before calling cross validate we will be breaking the golden rule by letting information from our validation data affect our preprocessing. 

To handle this problem and keep our code maintainable we can use pipelines. 

```python
scores = cross_validate(pipe, X_train, y_train, return_train_score=True)
```

Using a `Pipeline` takes care of applying the `fit_transform` on the train portion and only `transform` on the validation portion in each fold.

Pipelines allow us to define a series of transformers with a final estimator (predictor/model). 

## Imputation

We want to be able to replacing missing values with some reasonable values.

- For categorical values we may want to use the most frequent value
- In numeric columns we may want to use the mean or median of the column

## Numeric Features: Scaling

This problem affects a large number of ML methods. There are a number of approaches and we will look at two of the most popular. 

| Approach        | What it does                     | How to update X                                      | sklearn implementation                                                                                                                                 |
| --------------- | -------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| normalization   | sets range to [0, 1]             | `X-= np.min(X, axis=0)`<br/>`X /= np.max(X, axis=0)` | [`MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)                                          |
| standardization | sets sample mean to 0, s.d. to 1 | `X-= np.mean(X, axis=0)`<br>`X /= np.std(X, axis=0)` | [`StandardScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) |

## Categorical Features

Most algorithms require numeric inputs.

We can transform categorical features to numeric ones so that we can use them in the model.

- One-hot encoding - recommended in most cases
- Ordinal encoding - occasionally recommended

If we have a lot of categories we should ask ourselves:

>  Do we have enough data for rare categories to learn anything meaningful?

If we do not think we need all the categories we could try

- Grouping them into bigger categories
  - Example: country names into continents such as “South America” or “Asia”
- Or having “other” category for rare cases

### One-hot Encoding

Creates new binary columns to represent our categories. If we have c categories in our column, we create c new binary columns to represent those categories.

One-hot encoding is called one-hot because only one of the newly created features is 1 for each data point.

### Ordinal Encoding

Here we simply assign an integer to each of our unique categorical labels.

The encoder doesn’t know the order.

- We can examine unique categories manually, order them based on our intuitions, and then provide this human knowledge to the transformer.
- `X["ordinal_feature"].unique()` will give us all of the unique categories for our categorical data
- We can then make an ordered list, note that if you use the reverse order of the categories, it wouldn’t matter.
  - `class_attendance_levels = ["Poor", "Average", "Good", "Excellent"]`
- Ensure that we have all the same categories in our manual ordering
  - `assert set(class_attendance_levels) == set(X_toy["class_attendance"].unique())`
- Then we add them to the transformer

What’s the problem with this approach?

- We have imposed ordinality on the categorical data.
- In general, label encoding is useful if there is ordinality in your data and capturing it is important for your problem, e.g., `[cold, warm, hot]`.

## Text Features

How can we encode text data so that we can pass it into machine learning algorithms? Some popular representations of raw text include:

- Bag of words
- TF-IDF
- Embedding representations

### Bag of Words (BOW)

One of the most popular representation of raw text. Ignores the syntax and word order. Has two components:

1. The vocabulary (all unique words in all documents)
2. A value indicating either the presence or absence or the count of each word in the document.

In the Natural Language Processing (NLP) community text data is referred to as a **corpus** (plural: corpora).

Of course this is not a great representation of language, but it works surprisingly well for many tasks.

## Adding New Features

We can add new calculated features to our dataset, so long as they do not use any global information in the data - they must only look at other data in their row. Otherwise we are breaking the golden rule. 

* Note: if we are using features to calculate new features we may want to consider removing the old features.

### Picking Features

- Do you want to use certain features such as **gender** or **race** in prediction?
- Remember that the systems you build are going to be used in some applications.
- It’s extremely important to be mindful of the consequences of including certain features in your predictive model.
