# Project Overview
This project is the response to a Black Box Optimisation challenge as part of Imperial Business College Professional Certificate Program.  A Black Box challenge is an exercise to find the optimum (minimum or maximum) of an unknown (black box) function by submitting inputs and receiving outputs.  In this case, this sampling process is extremely limited, with only one sample per black box allowed per week.  There is no access to the objective function and the gradient is not available.  This replicates real-world situations where sampling the function is expensive or time consuming, forcing the use of a strategy to minimise the number of samples to find the optimum.

Given a limited number of initial samples, the approach is to use Bayesian Optimisation techniques to decide the next best sample location to maximise information retrieval about possible optimum location.

The challenge taken here is a total of 8 black box optimisation problems, ranging from 2 input dimensions up to 8.  This presents another challenge in terms of visualisation.  Most information explaining Bayesian optimisation presents the techniques in one dimension, which can be visualised in order to gain insight into the problem and intuitively understand the search strategy.  Multi dimension problems are not easy to visualise and therefore difficult to interpret the data from algorithms and the effect of hyper parameters.

The goal of this particular challenge is to gain working experience with the tools and techniques learnt on the program, in order to be able to apply them with confidence and relevance.  There are many tools and algorithms, and blind application of these algorithms will not yeild successful application.

# Inputs and Outputs
| Function | Goal | Input Dimensions | Input Range | Input Precision | Output |
|----------|------|------------------|-------|-----------|--------|
|1 | Maximise | 2 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|2 | Maximise | 2 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|3 | Maximise | 3 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|4 | Maximise | 4 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|5 | Maximise | 4 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|6 | Maximise | 5 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|7 | Maximise | 6 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 
|8 | Maximise | 8 |  $ 0 \leq d \leq 1$ | 6 decimal places | $ f:\mathbb{R}^d \to \mathbb{R}$ | 

The format of the input to the black box is each dimesion separated by '-'.  For example:
0.000000-0.000000-0.000000-0.000000

# Challenge Objectives
The objective of the challenge is to locate the inputs that yeild the global optimum of each of the 8 functions, given an initial set of samples. Each week one new sample can be submitted to the Black Box function to yeild an output.  Using this rresult to update prior belief, a new sample must be chosen to be submitted.  Chosing this new sample requires a strategy, deciding between exploration and explotation.  If the current belief as that the location of the optimum is close to current known maximum, then explotation of the space near the maximum could be chosen.  Alternatively, if there is a belief that the global optimum could be located in an unexpolred region of the problem space, then a decision needs to be made on where the best opportunity exists to potentially locate this optimum.  

# Technical Approach
1. Become proficient in Bayesian Optimisation and gain an understanding of Gaussian Processes as surrogate functions, and the use of acquisition functions to build a strategy of exploitation vs exploration for each of the functions.
2. Investigate visualisation tools to support building an intuition of setting a sampling strategy.
3. Acquire a number of data points to begen to form a picture of convergence or divergence (or no progress at all).
4. Over time, experiment with alternative optimisation algorithms to understand how they perform and whether there are better tools available.

## Week 1
- Apply default Bayesian Optimisation tools with default values.  Signficant error made in this round as default python tools optimise through minimisation.
## Week 2
- Investigate visualisation tools - PCA and Objective Plotting in each dimension.
- Use these plots to visulise the effects of verious hyperparameters, and attemp some intuition in selecting next sample
- Late submission means results of this round will not be availabe in time for week 3
## Week 3
- Continue with same strategy to build a more complete picture of the input space