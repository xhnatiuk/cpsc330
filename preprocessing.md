# Data Splitting

- Pass in X and y, or a dataframe containing both.

- Specify train or rest split sizes using `train_size` or `test_size`
  
  - Default is `test_size` = 0.25

- The `random_state` argument controls the data shuffling before the splitting.
  
  - We generally use this to ensure we can get the same results between runs.

```python
from sklearn.model_selection import train_test_split

# Using a dataframe
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
X_train, = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# Alternately,  Using X and y
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

## Cross Validation

`cross_val_score`: gives us a list of validation scores for each fold.

- It creates `cv` folds on the data (train/validation splits).
- In each fold, it fits the model on the training portion and scores on the validation portion.

Note that cross validation doesn't shuffle the data; that is done in `train_test_split`.

```python
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

# Pipelines

Allows us to define a pipeline of named transformers with a final estimator. Let’s combine the preprocessing and model with pipeline:

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

## `make_pipeline`

- Shorthand for `Pipeline` constructor
- Does not permit naming steps
- Instead the names of steps are set to lowercase of their types automatically; `StandardScaler()` would be named as `standardscaler`

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    SimpleImputer(strategy="median"), 
    StandardScaler(), 
    KNeighborsRegressor()
)
pipe.fit(X_train, y_train)
pipe.predict(X_train, y_train)
```

When you call `fit` on the pipeline, it carries out the following steps:

- Fit `SimpleImputer` on `X_train`
- Transform `X_train` using the fit `SimpleImputer` to create `X_train_imp`
- Fit `StandardScaler` on `X_train_imp`
- Transform `X_train_imp` using the fit `StandardScaler` to create `X_train_imp_scaled`
- Fit the model (`KNeighborsRegressor` in our case) on `X_train_imp_scaled`

Note that we are passing original data to `predict` as well. This time the pipeline is carrying out following steps:

- Transform `X_train` using the fit `SimpleImputer` to create `X_train_imp`
- Transform `X_train_imp` using the fit `StandardScaler` to create `X_train_imp_scaled`
- Predict using the fit model (`KNeighborsRegressor` in our case) on `X_train_imp_scaled`.

# Column Transformers

In most applications, some features are categorical, some are continuous, some are binary, and some are ordinal.

- We need to apply different transformations on different columns.

Steps:

1. Identify all the different types of data we want to perform transformations on
   - numeric, categorical, ordinal, binary/passthrough (don't do any transformations), drop (features to drop)
2. Identify the transformations we want to apply
3. Create column transformers

Each transformation is specified by a name, a transformer object, and the columns this transformer should be applied to.

```python
from sklearn.compose import ColumnTransformer
# apply scaling
numeric_feats = ["university_years", "lab1", "lab3", "lab4", "quiz1"]  
# apply one-hot encoding
categorical_feats = ["major"]  
ct = ColumnTransformer(
    [
        ("scaling", StandardScaler(), numeric_feats),
        ("onehot", OneHotEncoder(sparse=False), categorical_feats),
    ]
)
```

## `make_column_transformer`

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

- note that we can see what transformers are being applied using the `named_transformers_` attribute (`ct.named_transformers__`)

When we `fit_transform`, each transformer is applied to the specified columns and the result of the transformations are concatenated horizontally.

- A big advantage here is that we build all our transformations together into one object, and that way we’re sure we do the same operations to all splits of the data.
- Otherwise we might, for example, do the OHE on both train and test but forget to scale the test data.

Note that the returned object is not a dataframe. So there are no column names. How can we view our transformed data as a dataframe?

- We are adding more columns. So the original columns won’t directly map to the transformed data. Let’s create column names for the transformed data.

```python
column_names = (
    numeric_feats
    + ct.named_transformers_["onehotencoder"].get_feature_names().tolist()
    + passthrough_feats
)
pd.DataFrame(transformed, columns=column_names)
```

- Note that the order of the columns in the transformed data depends upon the order of the features we pass to the `ColumnTransformer` and can be different than the order of the features in the original dataframe.

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

# Imputation

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train_imp = imputer.transform(X_train)
X_test_imp = imputer.transform(X_test)
```

- Note that `imputer.transform` returns an `numpy` array and not a dataframe

# Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # create feature trasformer object
scaler.fit(X_train)  # fitting the transformer on the train split
X_train_scaled = scaler.transform(X_train) # transforming the train split
X_test_scaled = scaler.transform(X_test)  # transforming the test split
```

Alternatively we can use `fit_transform`, this combines the fit and the transform step. We can use this safely on our train data BUT NOT on our test data.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler() # or = MinMaxScaler
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
pd.DataFrame(X_train_scaled, columns=X_train.columns)
```

# Categorical Preprocessing

## One-hot Encoding

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
- If there are a huge number of categories, it may be beneficial to keep them sparse. For smaller number of categories, it doesn’t matter much.

### Unknown Categories & The Golden Rule

Sometimes if our dataset is small we can have data put into our validation split that then would have its own columns made by the transformers and are therefore not recognized during cross validation.

By default, `OneHotEncoder` throws an error because you might want to know about this. Simple fix:

- Pass `handle_unknown="ignore"` argument to `OneHotEncoder`
- It creates a row with all zeros.
- With this approach, all unknown categories will be represented with all zeros and cross-validation will run OK now.

If it’s some fixed number of categories, like provinces in Canada, we know the categories in advance. This is one of the cases where it might be OK to violate the golden rule and get a list of all possible values for the categorical variable.

### Binary OHE

Sometimes you have features with only two possible categories.

- If we apply `OheHotEncoder` on such columns, it’ll create two columns, which seems wasteful, as we could represent all information in the column in just one column with say 0’s and 1’s with presence of absence of one of one of the categories.
- You can pass `drop="if_binary"` argument to `OneHotEncoder` in order to create only one column in such scenario.

## Ordinal Encoding

We need to examine unique categories manually, order them based on our intuitions, and then provide this human knowledge to the transformer.

- `X["ordinal_feature"].unique()` will give us all of the unique categories for our categorical data

- We use these categories to make our ordered list
  
  - note that if you use the reveres order it doesnt make a difference

- Then we add them to the transformer

```python
# make manualr ordering
class_attendance_levels = ["Poor", "Average", "Good", "Excellent"]
# ensure that we have all the same categories in our manual ordering
assert set(class_attendance_levels) == set(X_toy["class_attendance"].unique())
# create transformer
oe = OrdinalEncoder(categories=[class_attendance_levels], dtype=int)
oe.fit(X[["class_attendance"]])
# get transformed column
ca_transformed = oe.transform(X[["class_attendance"]])
df = pd.DataFrame(
    data=ca_transformed, 
    columns=["class_attendance_enc"], 
    index=X.index
)
pd.concat([X, df], axis=1)
```

If we have more than one ordinal column we can pass the manually ordered categories as a list of lists.

- Manually create (and check) ordered categories for each ordinal feature

- Pass a list of lists to the transformer

# Text Preprocessing

## Bag Of Words

`CountVectorizer`

- Converts a collection of text documents to a matrix of word counts.
- Each row represents a “document” (e.g., a text message in our example).
- Each column represents a word in the vocabulary (the set of unique words) in the training data.
- Each cell represents how often the word occurs in the document.

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

- Most words do not appear in a given document.
- We get massive computational savings if we only store the nonzero elements.
- There is a bit of overhead, because we also need to store the locations:
  - e.g. “location (3,27): 1”.
- However, if the fraction of nonzero is small, this is a huge win.
- Code to check:

```python
print("The total number of elements: ", np.prod(X_counts.shape))
print("The number of non-zero elements: ", X_counts.nnz)
print(
    "Proportion of non-zero elements: %0.4f" % (X_counts.nnz / np.prod(X_counts.shape))
)
```

### `CountVectorizer` Hyperparameters

- `binary`: whether to use absence/presence feature values or counts
- `max_features`: only consider top `max_features` ordered by frequency in the corpus
- `max_df`: ignore features which occur in more than `max_df` documents
- `min_df`: ignore features which occur in less than `min_df` documents
- `ngram_range`: consider word sequences in the given range
