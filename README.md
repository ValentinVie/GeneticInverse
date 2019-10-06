# GeneticInverse
How to find the inverse of a square matrix using a Genetic Algorithm.

## Motivation
The exact inverse of a matrix can be easily computed using algorithms like Gaussian elimination. These algorithms run in O(n^3) which is not a problem in low dimension.

However for higher dimension, computing the exact inverse of a matrix isn't really practical. More, we might not need the exact inverse. An approximation of the inverse (relative to a norm) is enought for common applications, especially machine learning.

## Project

In this project I implemented a way to compute an approximate inverse of a matrix using a genetic algorithm with the Frobenius norm.

The genetic algorithm is obviously slower than the exact inverse for lower dimension but is actually quite accurate when enough time is given. 

I couldn't reach the dimensions where the genetic algorithm would actually be better (given a tradeoff on precision) than the classic inversion algorithm because of computing power limitations.