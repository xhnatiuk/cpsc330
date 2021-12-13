14: Clustering

# 15: Simple Recommender Systems

# 16: Text Data, Embeddings, and Topic Modeling

## NLP

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

# 17: Neural Networks and Computer Vision

# 18: Time Series Data

# 19: Survival Analysis
