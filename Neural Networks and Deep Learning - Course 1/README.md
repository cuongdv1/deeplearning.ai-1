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



# Vectorization

Vectorization is basically the art of getting rid of explicit loops in a program. This helps in reducing the amount of time taken to do some mathematical operations. 
As we saw in the code, the time taken to perform a simple mathematical operation using for loop and the the dot product has a huge difference. 



The rule of thumb in Vectorization is:

- Whenever possible, avoid explicit for-loops



# Vectorization of Logistic Regression Equations

As we saw earlier, the Hypothesis equation was given as,  $$ \hat{y} = \sigma (a*x + b)$$ where, the term inside the sigmoid function can be defined as, $$ z = a*x + b $$ 

If we consider a dataset, we'd have to consider multiple explicit for loops to get the required outputs. But here, we'll use vectorization methods to implement the equation for $z$ instead of using explicit for loops.

Thus, the vectorized equation becomes, 

$$ Z = [ z_1,z_2,z_3.....,z_m  ]  = w^T. X + B $$

where $$B$$ is: $$ B = [b_1,b_2,b_3,b_4,.......,b_m] $$

This can be implemented using NumPy as:

```python
Z = np.dot(w.T,X) + b
```

Here even though the value of b, is only one value, python NumPy converts it into a $1*m$ matrix. This process is called **Broadcasting** in Python. 

Hence, the equation of $\hat{y}$ becomes, 

$$ \hat{y} = \sigma(Z) $$

This completes the Vectorization of Hypothesis Equation. 

Now, lets vectorize the Gradient Descent Equation.

The change in the value of Z which is denoted as $ \delta z$ can be calculated using:

$$ \delta z = \hat{y} - Y $$

where $ Y $ is the actual output matrix.

Thus, we can calculate the changes in the weights and changes in the bias as:

$$ \delta w = \frac{1}{m} [X. \delta z^T] $$

$$ \delta b = \frac{1}{m}*np.sum(\delta z) $$

and the values of $w$ and $b$ can be calculated as:

$$ w := w - \alpha *\delta w $$

$$ b := b - \alpha * \delta b $$

Here $$\alpha$$ is the learning rate.



This completes the gradient descent in logistic regression. One interesting thing here that we'd have to note is that, here, we computed one iteration without using any explicit for loops. But if we have to perform multiple iterations, then we'd have to use a for loop which gives us the number of iterations.



# Broadcasting in Python

If we are performing some mathematical operation with matrices which don't have the same order, this method, broadcasting takes care of the order of the matrices to ease the performance of the mathematical operation. For example, if we want to add a column matrix with an integer value, the broadcasting method helps us here. If I want to perform the calculation as shown below, 

$$ \begin{bmatrix}1 \\ 2 \\3 \end{bmatrix} + 100  \ broadcasting \ converts \ this \ into $$ 

$$ \begin{bmatrix}1 \\ 2 \\3 \end{bmatrix} + \begin{bmatrix} 100 \\ 100\\ 100\end{bmatrix} resulting \ in \begin{bmatrix} 101\\102\\103\end{bmatrix} $$

Any kind of operation we perform, this is the kind of output we'd get. 

To generalize,

A matrix with the shape of m x n with some number whose shape is different, gets converted into an m x n matrix to perform the operation. 

