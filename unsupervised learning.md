# Clustering

Most of the data that exists is unlabeled. Getting labeled training data is often difficult, expensive, or simply impossible. 

The most intuitive way to extract useful information from unlabeled data is to group similar examples together to get some insight into the data. 

**Clustering** is the task of partitioning the dataset into groups called clusters. The goal of clsutering is to discovering underlying groups in a given dataset such that: 

* examples in the same group are as similar as possible

* examples in different groups are as different as possible

Clustering is based on the notion of similarity or distances between points. In a multi-dimensional space, we can use something like the k-neighbors for similarity. 

Usually clusters are identified by a cluster label. These labels are arbitrary and relabeling the points (label switching) does not make a difference. All we care about is which points have the same labels, and which ones have different labels. 

## Correct Grouping

Very often we do not know how many clusters there are in the data, or if there are any clusters at all. There is a notion of coherent and optimal (in some sense) cluster, but there is no absolute truth here.

Instead we look for meaningful groups. Meaningful groups are dependent on the application. It helps to have prior knowledge about the data and the problem. 

This makes it hard for us to objectively measure the quality of a clustering algorithm. 

## Common Applications

Although there is no notion of the "right" answer, we might still get something useful out of clustering. There are a number of common applications for clustering.

Customer segmentation: Understand landscape of the market in businesses and craft targeted business or marketing strategies tailored for each group.

Document clustering: Grouping articles on different topics from different news sources.

### Data Exploration

* Summarize or compress data

* Partition data into groups before further processing

For instance, we may use it in tandem with supervised learning. We could carry out a clustering and examine the performance of our model on different clusters. If the performance is lower on a particular cluster, we could build a separate model for that cluster and create an ensemble, improving the overall performance of our model. 

# 15: DBSCAN & Simple Recommender Systems

Most often the data for recommender systems come in as ratings for a set of items from a set of users. 

Two entites:

1. N users

2. M items

A utility matrix is a matrix that captures interactions between N users and M items. An interaction can come in many different forms (clicks, ratings, purchases etc...)

Sparse Utility Matrix: users only interact with a few items so the matrix is very sparse.

With recommender systems we are tryign to comptlete the utility matrix, or predict the missing values. 

Recommendation systems problem is a**matrix completion problem**.

## Collaborative Filtering

One of the most popular approaches to recommendation systems.

* Unsupervised: only uses the user item interactions given in the ratings matrix

Intuition: we may have similar users and similar items which can help us predict missing entries. We can leverage social information to provide recommendations.

## Content Based Filtering

Supervised learning approach. In collaborative filtering we assumed that we only have the ratings data. Usually there is some information on items and users availiable. We can use this information we predict ratings in the utility matrix.

Hybrid Filtering: combines the advantages of collaborative filtering and content-based filtering. 

# 16: Introduction to Natural Language Processing

What is natural language processing?

- Search engines use natural language processing to understand your queries

Natural language processing is the process of understanding what humans are saying or writing.

It has different names depending on which part of the problem you are focusing on:

- Engineering: Natural Language Processing

- Language Understanding: Computational Linguistics

- Field: Human Language Processing or Speech and Language Processing

There are some differences between these fields but they have a very large overlap and tend to use the same methods.

Applications of NLP:

- Voice Assistants

- Smart Compose

- Translation

Challenges:

- Language is complex and sublte

- Language is ambiguous at different levels

- Language understanding involves common-sense knowledge and real-world reasoning

- All the problems related to representation and reasoning in AI arise from this domain

Types of Ambiguity:

- Lexical: what does a word mean

- Referential: what does a pronoun refer to

## Word Embeddings

The idea is to represent  word meaning so that similar words are grouped together. 

Modeling word meaning that allows us to draw useful inferences to solve meaning-related problems and to find relationships between words. 

We need a representation that captures the relationships between words. We will be looking at two:

1. Sparse representation with term-term co-occurence matrices

2. Dense representation with Word2Vec

Both are based on the ideas of: 

* distributional hypothesis 
  
  * we know something about a word by the other words around it

* vector space model
  
  * the meaning of a word can be modeled by placing it in a vector space
  
  * we create embeddings of words so that teh distances among words in the vector space represent the relationships between them (closer words are more similar)

### Term-Term Co-Occurence Matrix

Document-term co-occurence matrix: Bag-of-words representation of text. 

We can also do this with words. The idea is to go through a large corpus of text and keep a count of all the words that appear in context of each word within a window

The similarity is calculated using dot products between word vectors. 

- Higher the dot product more similar the words.

We are able to capture some similarities between words now.

Term-term co-occurrence matrices are long and sparse (most of the elements are 0). This is okay because there are efficient ways to deal with sparse matrices.

### Word2Vec

A family of algorithms to create dense word embeddings

Alternatively we can try and learn short and dense vectors:

- these may be easier to train ML modes with (less weights to train)

- they may also generalize better

- They work better in practice!

Word2Vec is able to capture complex relationships between words. 

Instead of training our own models, we use the **pre-trained embeddings**. These are the word embeddings people have trained embeddings on huge corpora and made them available for us to use.

### Implicit Biases and Stereotypes in Word Embeddings

Word embeddings reflect the gender stereotypes present in broader society. They may also amplify these stereotypes because of their widespread usage. 

Luckily most of the modern embeddings are de-biased.

## Topic Modelling

Topic modeling gives you an ability to summarize the major themes in a large collection of documents (corpus). 

- A common tool to solve such problems is unsupervised ML methods.
- Given the hyperparameter K, the idea of topic modeling is to describe the data using K “topics”

Input:

* large collection of documents

* K

Output:

* Topic-words association: for each topic what words describe that topic?

* Document-topics association: for each document: what topics are expressed by the document?

## Text Preprocessing

Text data is unstructured and messy. If we want to use it in a model we need to 'normalize' it.

**Lemma**: Words with the same stem, same part-of-speech, roughly same meaning

Types vs Tokens:

- Type: an element in the vocabulary

- Token: an instance of that type in text

Types are task dependent. In some tasks we may consider UBC and The University of British Columbia to be one type, in others it may be two.

### Tokenization

Preprocessing is often task specific. But Tokenization is used in almost all NLP preprocessing.

This is generally the first step in a machine learning pipeline, and if there are errors here they will propogate all the way through our modeling.

Tokenization is a two step process:

1. Sentence segmentation: split text into sentences

2. Word tokenization: split sentences into words

#### Sentence Segmentation

In english the period (.) is quite ambiguous. The ! and ? are also relatively ambigiuous. This is not the case in all languages; in Chinese the period is unambiguous.

We could write regular expressions. But there are good off-the-shelf models for this task.

That being said, if your data is really messy, you may have to do some postprocessing to fix the sentence segmentation after using a pre-trained segmenter.

#### Word Tokenization

The process of identifing word boundaries is referred to as tokenization.

- **Type** an element in the vocabulary

- **Token** an instance of that type in running text

How many words are there in a sentence? Is whitespace a sufficient condition for a word boundary?

This depends on our definition of a word. For example:

- Is British Columbia one word or two words?

- Is punctuation a word?

- What about the punctuations in abbreviations? (like U.S.)

- What should we do with words like Master's?

Again we could use regular expressions. But there are good off-the-shelf models for this task.

##### Word Segmentation

In some languages we need much more sophisticated tokenization.

- For languages such as Chinese, there are no spaces between words.

- For German, compound words are not seperated.

There are models for these tasks as well.

### Other Common Preprocessing Steps

#### Punctuation and Stopword Removal

The most frequently occurring words in English are not very useful in many NLP tasks. Generally these are propositions or determiners. For example: 'the', 'is', 'a', and punctuation.

Because they are not very informative in many tasks we often remove them during the preprocessing.

#### Stemming and Lemmatization

For many NLP tasks we want to ignore the morphological differences between words. Lemmatization converts inflected forms into the base form.

- Ex: studying, studied -> study

Stemming is an aggressive crude chopping of affixes.

- Ex: automates, automatic, automation -> automat

Usually these reduced forms (stems) are not actual words. Generally renders the text unreadable. Be careful using stemming. Think about if this is actually what you want to do to your text.

#### Other

To understand our text, we often want to extract more information that just lemmas. Some other common tasks in NLP pipelines include:

- Part of Speech Tagging: assigning part-of-speech tags to all words in a sentance

- Named Entity Recognition: labelling named 'real world' objects like people, companies, or locations

- Coreference Resolution: deciding whether two strings refer to the same entity (UBC vs University of British Columbia)

- Dependency Parsing: Representing the grammatical structure of a sentence

# 17: Multi-Class Classification and Computer Vision

Many linear classification models don’t extend naturally to the multiclass case. A common technique is to reduce multiclass classication into several instances of binary classification problems.

Two kind of “hacky” ways to reduce multi-class classification into binary classification:

- the one-vs.-rest approach

- the one-vs.-one approach

## One-Vs-Rest

Learn a binary model for each class which tries to separate that class from all of the other classes. If we have k classes, we train k binary classifiers (one for each class).

* Train on imbalanced datasets containing all examples

* Given a test point, get scores from all the binary classifiers

* The classifier with the highest score wins and that is the predicition

## One-Vs-One

Build a binary model for each pair of classes.

* Trains $\frac{n(n-1)}{2}$ binary classifiers

* Trained on relatively balanced subsets

To predict we apply all of the classifiers on a test example, count how often each class was predicted, and predict the class with the most votes. 

## Computer Vision

Refers to understanding images/videos, usually using ML/AI. Tasks of interest include:

- mage classification: is this a cat or a dog?

- object localization: where are the people in this image?

- image segmentation: what are the various parts of this image?

- motion detection: what moved between frames of a video?

Neural Networks are very popular these days under the name deep learning. The apply a sequence of transformations on our input data. At a very high level we can think of them as a pipeline. 

- They can be viewed a generalization of linear models where we apply a series of transformations.

- They can learn very complex functions.
  
  - The fundamental tradeoff is primarily controlled by the **number of layers** and **layer sizes**.
  
  - More layers / bigger layers –> more complex model.
  
  - You can generally get a model that will not underfit.

Pros:

- they work really well for structured data:
  
  - 1D sequence, e.g. timeseries, language
  
  - 2D image
  
  - 3D image or video

- They’ve had some incredible successes in the last 10 years.

- Transfer learning is really useful.

Cons:

- When you call `fit`, you are not guaranteed to get the optimal.
  
  - There are now a bunch of hyperparameters specific to `fit`, rather than the model.
  
  - You never really know if `fit` was successful or not.
  
  - You never really know if you should have run `fit` for longer.

### Transfer Learning

In practice, very few people train an entire CNN from scratch because it requires a large dataset, powerful computers, and a huge amount of human effort to train the model.

- Instead, a common practice is to download a pre-trained model and fine tune it for your task. This is called **transfer learning**.
- Transfer learning is one of the most common techniques used in the context of computer vision and natural language processing.

# 18: Time Series Data

**Time series** is a collection of data points indexed in time order.

Issues: 

* If we split normally we may be training on data that came after our test data. If we are trying to forecase we cannot know what happened in the future! 

* Tree-based models cannot *extrapolate* to feature ranges outside the training data.

* Cannot do normal cross validation (we will be predicting the past)
  
  * There is [`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) for time series data.

Feature Engineering for date/time Columns

* Split time series into multiple numeric features: year, month, day of week, hour

This works for tree based models, but not for ridge, for ridge we need to encode these features as categorical data.

* we may also want to add interaction features (i..e Saturday 9:00pm, monday 11:00pm etc...)

Lagged feature: what if tomorrows datapoint is related to today and yesterday? 

## Forecasting Into the Future

Say we want to predict 7 days into the future instead of one day. Main approaches:

1. Train a separate model for each number of days. E.g. one model that predicts Tomorrow, another model that predicts n2Days, etc. We can build these datasets.

2. Use a multi-output model that jointly predicts Tomorrow, in2Days, etc.  outside the scope of CPSC 330.

3. Use one model and sequentially predict using a `for` loop. However, this requires predicting *all* features into a model so may not be that useful here.
   
   1. Predict Tuesday’s sales
   
   2. Then, to predict for Wednesday, we need to know Tuesday’s sales. Use our *prediction* for Tuesday as the truth.
   
   3. Then, to predict for Thursday, we need to know Tue and Wed sales. Use our predictions.

## Trends

If we use **linear regression** we’ll learn a coefficient for `Days_since`.

* If that coefficient is positive, it predicts unlimited growth forever. That may not be what you want? It depends.
- If we use a **random forest**, we’ll just be doing splits from the training set, e.g. “if `Days_since` > 9100 then do this”.
  
  - There will be no splits for later time points because there is no training data there.
  
  - Thus tree-based models cannot model trends.
  
  - This is really important to know!!

- Often, we model the trend separately and use the random forest to model a de-trended time series.

# 19: Survival Analysis

Imagine that you want to analyze *the time until an event occurs*. For example,

- the time until a disease kills its host.

- the time until a piece of equipment breaks.

- the time that someone unemployed will take to land a new job.

- the time until a customer leaves a subscription service (this dataset).

Although this branch of statistics is usually referred to as **Survival Analysis**, the event in question does not need to be related to actual “survival”. The important thing is to understand that we are interested in **the time until something happens**, or whether or not something will happen in a certain time frame.

Frequently, there will be some kind of **censoring** which will not allow you to observe the exact time that the event happened for all units/individuals that are being studied.

* This is right censoring

Solutions:

* Only consider examples where we have the result (i.e. churn = yes, dead = yes, etc...)
  
  * On average they will be **underestimates** (too small), because we are ignoring the currently subscribed (un-churned) customers. Our dataset is a biased sample of those who churned within the time window of the data collection. Long-time subscribers were more likely to be removed from the dataset! This is a common mistake - see the [Calling Bullshit video](https://www.youtube.com/watch?v=ITWQ5psx9Sw) 

* Assume everyone churns/dies/etc right now
  
  * It will be an **underestimate** again. For those still subscribed, while we did not remove them, we recorded a total tenure shorter than in reality, because they will keep going for some amount of time.

* Survival analysis

## Kaplan-Meier survival curve

Doesnt look at features

- Interpret a survival curve, such as the Kaplan-Meier curve.

## Cox proportional hazards model

Similar to linear regression

- Interpret the coefficients of a fitted Cox proportional hazards model.
