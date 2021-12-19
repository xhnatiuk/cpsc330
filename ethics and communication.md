# Ethics

The data we collect and use has consequences either directly or indirectly for people. These consequences may be regariding their privacy, livelihood, freedom, or identity.

Collecting data obscures these consequences between something that seems more scientific and abstract.

The golden rule of ethics: treat other's data as you would want yours treated!

## Experimental Design

Many experiements with human subjects are not run by academics, who are subject to regulations, but are rather ran by tech companies, who do so at scale with no regulations. These unregulated experiments are not subject to the review of external committees and do not require consent.

Randomized human subject experiements, called A/B tests, are happening all the time on the web.

Experimental design goes beyond experiments that are performed on humans. A huge amount of data is also being passively collected about you. This data is often used in observational studies. This information is often used in public health applications to better understand people and their health. However it is worth considering the potential risks of re-purposing unconsented data from millions of people, and how we can minimize risk of consequences to these individuals.

## Data Collection

### Sampling Bias

An inportant concept in ML is that training data is sampled from a population and a predictor is developed based on that sampling. Both the population being sampled and the sampling scheme can have major impacts on the results of the algorithm.

This issue is often overlooked by engineers who dont understand the sources of variation in their data.

### Privacy

The exponential decrease in the price of collecting and storing data has lead to an entire underground economy revolving around the data collected about you. Virtually everything collects information about you and lots of it is bundled, bought, and sold by data brokers. It is frequently used by researchers, businesses and others trying to understand human behavior for various purposes, such as studies or advertising.

It is difficult to imagine a future where data is not collected at large scales. Increasingly, the collection and use of data is being regulated.

- The General Data Protection Regulation: applies to citizens of the EU and outlines regulations regarding the collection and use of peopleas' data.

- California Consumer Privacy Act : california's privacy regulations

These laws unfortunately only apply to individuals in these geographic regions, and do not out the use of personal information.

### Surveillance

There are parts of the world where facial recognition is used extensively to monitor the behavior of individuals. Drones have been used to conduct warrantelss surveillance of residents.

Credit scores are based on finnacial transaction data passively collected from credit cards, loan applications, and bank accounts. This data may be used to decide who has access to financial services.

## Objective Functions

[Goodhardt’s Law](https://en.wikipedia.org/wiki/Goodhart%27s_law), as stated by Marilyn Strathern in its general form is:

> When a measure becomes a target, it ceases to be a good measure.

A key issue in data science ethics is in the objective functions we choose to optimize.

The simplification of complex processes into a small number of necessarily limited objectives can have unintended consequences.

Often we will be encouraged to identify metrics and targets, and to optimize our data collection, algorithms, and inference, to optimize for these objectives. It is important to consider the consequences of narrow optimization.

#### Varying Objectives

It is important to understand what 'good enough' is in different contexts. While I may be okay with an algorithim that sometimes labels the wrong individual in a photo, I am unlikely to be okay with an autonomous vehicle that makes a similar amount of mistakes.

Objective functions simplify data science. This can lead to bias from not considering alternatives to our objective function, and by choosing how the function is optimized.

Moving to quantative definitions for objective functions allows us to clearly explain what we are optimizing and in turn allows us to consider the consequences of that optimization.

## Algorithims

Algorithims are increasingly being used for more and more important decisions. Many of the ethical issues in algorithms can be traced to sampling issues, or the choice of objective function.

However, algorithims have their own ethical issues.

They often give the appearance of being objective, without being objective. People tend to assume that algorithms are more objective than humans. This allows people to use highly biased or unfair algorithms as objective. The objective function, the data, the choices made by an analyst, can all have a major impact on the results of any algorithm.

Algorithims are not designed to understand nuance. This can be mildly frustrating when using a chatbot but detrimental for people tiraged by algorithms in hospitals or denied credit.

Algorithms are difficult to understand and rarely audited. They can also not be availiable for study due to the value of the intellectual property.

We can use model cards to partially address this: a brief summary of the data, algorithm, and applications of a model, including populations to which it can be fairly applied. This is an example of interpretable ML which is growing in popularity.

## Interfaces

It is important to consider how the human on the other side will be interacting with an interface you are optimizing.

> Dark patterns are user interface design choices that benefit an online service by coercing, steering, or deceiving users into making unintended and potentially harmful decisions.

This [dataset](https://webtransparency.cs.princeton.edu/dark-patterns/) has a collection of such dark patterns across thousands of websites.

#### The Exchange

A core exchange at the heart of our current technological revolution goes like this:

> “I will give you something for free in exchange for your data, which I will then sell to a third party to make revenue.”

This business model drives most social media companies, search companies, and a variety of other smaller technology startups that you may or may not know about.

These companies build technologies that help people do things that they want to do and give it away for free. But the consequence is that their data becomes the product that is sold. This has lead to the [famous saying](https://slate.com/technology/2018/04/are-you-really-facebooks-product-the-history-of-a-dangerous-idea.html):

> “If you are not paying for it, you’re not the customer; you’re the product being sold.”

## Decision Making

You willl often work with people who have a less experience analyzing data than you do. This is a challenge since the implications of data or algorithims can be subtle and difficult to explain.

These decisions can also be influenced because your manager may have a motivation to get a specific answer or re-purpose the data in a way that is compromising. This can be difficult since you may have to choose between doing what is right and being fired.

People in positions of power may choose to use data science in ways that introduce unfairness or bias. These issues may be clear to you but unclear to those without technical insights.

Sometimes data is used as a prop to justify discriminatory, unethical, or unfair decisions. You must consider the implications of your work more broadly for society, in a era where data is so prevalent.

## What You Should Do

Simplified advice is unlikely to generalize well across cases. Each data analysis requires its own careful consideration of its risks and benefits.

# Communicaiton

Why does communication matter? 



## Graphs

Graph literacy is like any other type of literacy. It is a skill that we can get better at.

Graphs have 5 ways of misleading you:

- Poor design

- Dubious Data

- Insufficient Data

- Concealing Uncertainty

- Suggesting Misleading Patterns

Designing good graphs is hard. Every decision in the design affects the story they tell, and how clearly they tell it.

A good design needs to strike a balance between simplicity and accuracy. As a reader, you need to ensure that the design choices compliment the data, rather than exaggerate it.

That being said, the design doesnt matter if teh data is bad or dubiously sourced. If a graph doesnt have a known source, its just a picture. To avoid falling for bad data, you need to check that the data source exists, is credible, and is relevant. You should also check that the data and the graph are telling the same story - this involved making sure that the graph unbiased and is not cherry picking statistics or burying an important point in a bunch of irrelevant numbers.

Unvertainty is inevitable, so you need to entertain the possiblity that the numbers in your data may not necessarily be true.

A graph can suggest misleading patterns:

> Dont read too much into a chart, particularly if you're reading what you would like to read.
