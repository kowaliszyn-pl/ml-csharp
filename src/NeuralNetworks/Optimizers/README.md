# Optimizers

## Stochastic Gradient Descent (SGD)

## Stochastic Gradient Descent with Momentum

File: [`StochasticGradientDescentMomentum.cs`](StochasticGradientDescentMomentum.cs)

### Description

> With Stochastic Gradient Descent we don’t compute the exact derivate of our loss function. Instead, we’re estimating it on a small batch. Which means we’re not always going in the optimal direction, because our derivatives are ‘noisy’. So, exponentially weighed averages can provide us a better estimate which is closer to the actual derivate than our noisy calculations. This is one reason why momentum might work better than classic SGD.

> The other reason lies in ravines. Ravine is an area, where the surface curves much more steeply in one dimension than in another. Ravines are common near local minimas in deep learning and SGD has troubles navigating them. SGD will tend to oscillate across the narrow ravine since the negative gradient will point down one of the steep sides rather than along the ravine towards the optimum. Momentum helps accelerate gradients in the right direction.[^tds]

[^tds] https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d