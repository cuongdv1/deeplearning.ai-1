# Binary Classification

In Machine Learning, there are two different types of algorithms. They are:

1. Regression Algorithm
2. Classification Algorithm



In **Regression Algorithm,** the target or the output is a continuous valued output where as in **Classification Algorithm**, the target is discrete. 

Through out this course, we will be looking at mainly Classification Algorithms. 

The most basic type of classification algorithms are the **Binary Classification Algorithms**. In Binary Classification algorithm, the output will have two discrete values. The discrete values may be text values, numbers, or Booleans.

A traditional example of Binary Classification algorithm is classifying images of cats and dogs. Another example can be classifying an email, as a spam or not spam email. 



We will be using Neural Networks to work on Binary Classification. 



The execution of Neural Networks takes place in two steps which are named as:

1. Forward Propagation
2. Backward Propagation



In **Forward Propagation**, the neural network moves forward over the complete data set once, and in **backward propagation**, the neural network moves back from the output to the input once. This two steps put together forms **one epoch**.



To implement Binary Classification, we will build an algorithm called as Logistic Regression



# Logistic Regression

Implementation of Logistic Regression mainly requires 3 different equations. These equations are generally called as:

1. Hypothesis Equation
2. Cost Function
3. Gradient Descent



The hypothesis equation is represented as:

$$ \hat{y} = \sigma (a*x + b)$$

The cost function is represented as:

$$ J(a,b) = -\frac{1}{m}\sum_0^m[y*ln(\hat{y}) + (1-y)*ln(1-\hat{y})]$$

The partial derivative of the cost function with respect to the cost function gives us the gradient descent along the a axis and b axis. These equations become

$$ \frac{\delta J(a,b)}{\delta a} = -\frac{1}{m} [\hat{y} - y]*x$$

$$ \frac{\delta J(a,b)}{\delta b} = -\frac{1}{m} [\hat{y} - y]$$

The values of the weights i.e. **a** and the bias **b** are varied using these gradient values. 

$$ a = a - \alpha[\frac{\delta J(a,b)}{\delta a}] $$

$$ b = b - \alpha[\frac{\delta J(a,b)}{\delta b}] $$



Usually, the value **a** are called as the **weights of the neural network** and the value **b** is called as the **bias in the neural network**.

Neural networks mainly work on the basis of varying the values of the weights and the bias. 

This is the general overview of Neural Networks.