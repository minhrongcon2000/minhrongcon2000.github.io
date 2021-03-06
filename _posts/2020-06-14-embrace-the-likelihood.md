---
layout: post
author: "Pham Hoang Minh"
title: "Embrace the likelihood"
permalink: "/:title"
categories: statistics likelihood bayesian frequentist
---
# Introduction
The very first machine learning model that one would construct when starting his or her journey in the field of Machine Learning is Linear Regression, an algorithm to find the best linear model to fit a data set. In Machine Learning approach, Linear Regression has five main steps: 
* Initialize a weight vector
* Pass the feature matrix to the model
* Calculate the mean-squared-error (MSE) loss function:

$$
\begin{align}\label{eq:1}
	J(w) = \frac{1}{2m}\sum_{i=1}^m (h_w(x) - y)^2 
\end{align}
$$

* Calculate its gradient vector $\vec{\nabla} J$
* Update the weight by using gradient descent to minimize the loss function.

However, the way to derive the loss function is usually ommitted. It turns out to be a very interesting topic that most of the statistical textbook have overlooked: the difference between likelihood and probability.

# Probability vs Likelihood: Frequentist vs Bayesian
There is an interesting fact that there was a conflict between these two ideologies. Frequentist was popular in the 18th-19th century since mathematics of this time is purely based on theory. Bayesian approach started gaining popularity recently since a lot of applied mathematical model in the modern day requires the incorporation of uncertainty in it.

Frequentist approach is, in fact, the very first thing that a beginner in Probability and Statistics would tackle. At the high level, Frequentist approach emphasizes on the proportion of occurrence of events without taking any outside effect into account. For example, in Frequentist approach, the chance that a fair coin landing on head is 1/2 since there are two possible outcomes: head and tail, and landing on each one of them is equally likely to take place. At the low level, Frequentist apprach belives that the chance of occurrence of an event is defined as follow.

$$ P(A) = \lim_{n\to\infty}\frac{n(A)}{n} $$

where $n(A)$ is the number of times the event $A$ happens in the total of $n$ trials.

Frequentist calls this chance probability. Frequentist approach has been a fundamental building block for various theory in Statistics, one of which is probability distribution such as the famous Gaussian (or so-called normal) distribution, and the Binomial distribution (which can be considered a discrete version of the former).

The weakness of Frequentist approach is that it does not take external information into account. A coin in Frequentist approach is always a fair coin, which never exists in the real world scenario. Bayesian approach is derived to tackle the issue. Taking the coin tossing problem as an example, in Frequentist approach, its probability of landing on head or tail is assigned to be 1/2. In Bayesian approach, however, the chance that the coin lands on head is $p$, and lands on tail is $1-p$, where $p$ is a random variable. Bayesian approach finds the optimal $q$ based on the data of a certain number of coin tossing. For instance, a coin is tossed 10 times, in which it lands on head 7 times. Frequentist approach would judge that the sample size is not sufficiently large, and would result in doing exhaustive experiments. Bayesian approach assumes that we can only draw that limited amount of sample, and perform inference on $p$. 

Let $A$ be the event that a coin lands on head 7 times out of 10 tosses. Since $p$ is a random variable, it follows some kind of probability distribution in Frequentist approach. In Bayesian approach, we define a prior belief (or a prior probability distribution) on $p$, so we assume that $p$ follows a uniform distribution between 0 and 1. Then, by using the Bayes theorem,

$$
\begin{align}\label{eq:2}
	f(p|A) = \frac{f(A|p)\pi (p)}{\int_0^1 f(A|p)\pi (p)dp}
\end{align}
$$

where, in Bayesian terms, $f(p\|A)$ is the posterior distribution of $p$ given $A$, $f(A\|p)$ is the likelihood that the event $A$ will occur given the probability of landing on head is $p$, $\pi (p)$ is the prior distribution of $p$. 

In Frequentist approach, given that the probability of landing on head is $p$, the chance of $A$ follows a binomial distribution with the proportion of success is $p$; thus, $f(A\|p) = \binom{10}{7}p^7(1-p)^3$. Since we assume that $\pi (p)$ is a uniform distribution between 0 and 1, $\pi (p)=1,\text{ }\forall p\in [0,1]$. Since the denominator of (\ref{eq:2}) is always a constant for all p in $[0,1]$, we can rewrite (\ref{eq:2}) as 

$$
\begin{align}
	f(p|A) = C\times f(A|p)\times\pi (p) = K\times p^7(1-p)^3
\end{align}
$$

where $C$ and $K$ are constants.

Our goal is to find $p$ such that the occurrence of the event is maximize. In other words, given the event $A$ happened, we want to find $p\*$ such that $f(p\|A)$ reaches its global maxima. Using Calculus approach, the optimal $p$ in this case is $0.7$.

In comparison to Frequentist approach, Bayesian approach introduces a new concept in conducting inferences: likelihood. Frequentists assume that the chance of an event happening is derived through repeated experiments and call it probability, whereas Bayesians believe that the chance of an event happening is also a random variable following a probability distribution, and finds the optimal chance of occurence through the data drawn from experiment with specifying their prior belief over the occurence chance. Sometimes, prior belief in Bayesian approach does have a major effect in conveying inference on the considered random variable. The methods performed above is called Maximum Likelihood Estimation, and it has been widely used in a lot of statistical textbooks without explaining its origin. It is the basics to derive the mean-squared-error loss function for linear regression.

# From Maximum Likelihood Estimation (MLE) to Linear Regression

We consider a general regression problem as follows: suppose that a vector of random variable $\vec{X}$ and a random variable $Y$ have the relationship of $Y=f(\vec{X})$ where $f$ is a continuous function. To find $f$, we draw a lot of samples of $\vec{X}$ and $Y$ from their distribution, and perform inference on the drawn samples, which is exactly the Bayesian apprach. We denote $\mathcal{D} = \\{(\vec{x_i}, y_i) \| i=1,2,\ldots,m\\}$ as the set of all drawn samples, where $\vec{x_i}\in\mathbb{R}^n$ (n is so-called the number of features), $y_i\in\mathbb{R}^n$, and $y_i = f(x_i) + \epsilon_i$, and $\epsilon_i\sim p(\epsilon)$.

The common approach to this problem is, first, to assume that $f$ is a member of a family of function $\mathcal{F}$, and second, to find the optimal $f^\*\in\mathcal{F}$ that best fits $\mathcal{D}$. In order to define the criteria to evaluate how well the function $f$ fits $\mathcal{D}$, the maximum likelihood estimation (MLE) method is used. 

Our goal is to maximize the likelihood 

$$
\begin{align}\label{eq:3}
P(Y|X) = \prod_{i=1}^m p(y_i|\vec{x_i}) = \prod_{i=1}^m p(\epsilon_i)
\end{align}
$$

since knowing the appearance of $\vec{x_i}$ will result in the prediction $\hat{y_i} = f(x_i)$, from which $\epsilon_i$ is calculated.

The mean-squared-error is derived from the assumption that $\epsilon\sim\mathcal{N} (0, \sigma^2)$. With that assumption, by taking natural logarithm both sides (\ref{eq:3}), it can be rewritten as 

$$
\begin{align}\label{eq:4}
\ln P(Y|X) = C - \sum_{i=1}^m \frac{(y_i - f(\vec{x_i}))^2}{2\sigma^2}
\end{align}
$$

where $C$ is a constant.

By maximizing $(\ref{eq:4})$, the term $J = \displaystyle{\sum_{i=1}^m} (y_i - f(\vec{x_i}))^2$ is also minimized (since $\sigma^2$ is assumed to be a constant), which is the famous mean-squared-error in Linear Regression. 

From here, one may ask whether another loss function is derived by using this approach, by making different assumption on the distribution of $\epsilon$. The answer is yes; in fact, there is a kind of loss function starting to gain popularity in Linear Regression called the mean-absolute error

$$
\begin{align*}
J = \sum_{i=1}^m |y_i - f(\vec{x_i})|
\end{align*}
$$

by assuming that $\epsilon$ follows Laplace distribution with mean 0. The only constraint in our assumption here is that the mean of the distribution to be assumed must be 0 (if the mean is not zero, we can shift the distribution to the left or the right based on the positivity or negativity of the mean).

With this approach, the cross-entropy loss function of Logistic Regression can also be derived. The only difference is that Logistic Regression assume its model as $g(\vec{x_i}) = \frac{1}{1+\exp\\{-f(\vec{x_i})\\}}$, where $f$ is called the boundary decision. The derivation is left for the reader, and the derivation $g$ will be ommitted since it is out of the scope of this blog.

# Conclusion
The title of this blog is to emphasize to the role of likelihood in the field of Machine Learning, a concept that has been overlooked in a lot of basic statistical material. This blog has clarified the difference between Frequentist and Bayesian approach, and has linked a lot of familiar concept in Statistics to Bayesian one. It also shows an application in one of the most popular, yet fundamental algorithm in the field of Machine Learning, in general, and of Deep Learning, in specific, which provides a building block for the derivation of loss function for new model development.