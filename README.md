# minSQN : Stochastic Quasi-Newton Optimization in MATLAB

Authors: [Nitish Shirish Keskar](http://users.iems.northwestern.edu/~nitish/) and [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home)

Please contact us if you have any questions, suggestions, requests or bug-reports.

## Introduction

This is a package for solving an unconstrained minimization
problem of the form,
min f(x) = (1/n)*sum_i f_i(x).

minSQN allows for the user to solve large-scale (sum-of-functions)
optimization problems using one of 11 Stochastic Quasi-Newton methods.



The following table summarizes all the methods that minSQN contains. The
methods are classified in terms of:
- Hyperparameters
- Length of LBFGS memory (limited/inf) [if inf, BFGS method used]
- Powell damping
- Hessian damping
- Curvature pair update (y)

```
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|  Method  | Hyperparameters | LBFGS Memory | Powell Damping | Hessian Damping |     Curvature pair     |          Reference          |
|          |                 | (finite/inf) |      (Y/N)     |      (Y/N)      |       (y) update       |                             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|    SQN   |     alpha, L    |    finite    |        N       |        N        | Hessian-vector product |     Byrd et. al. (2014)     |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|   DSQN   |     alpha, L    |    finite    |        Y       |        N        | Hessian-vector product |              --             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|   oBFGS  |   alpha, delta  |      inf     |        N       |        Y        |  Gradient differencing |  Schraudolph et. al. (2007) |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|  oLBFGS  |   alpha, delta  |    finite    |        N       |        N        |  Gradient differencing | Schraudolph et. al. (2007), |
|          |                 |              |                |                 |                        |   Mokhtari et. al. (2014)   |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|  D-oBFGS |   alpha, delta  |      inf     |        Y       |        Y        |  Gradient differencing |              --             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
| D-oLBFGS |   alpha, delta  |    finite    |        Y       |        N        |  Gradient differencing |              --             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|    RES   |   alpha, delta  |      inf     |        N       |        Y        |  Gradient differencing |   Mokhtari et. al. (2014)   |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|   L-RES  |   alpha, delta  |    finite    |        N       |        N        |  Gradient differencing |                             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|  SDBFGS  |   alpha, delta  |      inf     |        Y       |        Y        |  Gradient differencing |     Wang et. al. (2014)     |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
| L-SDBFGS |   alpha, delta  |    finite    |        Y       |        Y        |  Gradient differencing |              --             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
|   adaQN  |     alpha, L    |    finite    |        N       |        N        | Hessian-vector product |    Keskar et. al. (2015)    |
|          |                 |              |                |                 |   accumulated Fisher   |                             |
+----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
```

### Features
The minSQN package

* is written in pure-MATLAB with minimal dependencies and emphasizes simplicity, extendibility and cross-platform compatibility. 
* allows the user to run 11 different stochastic quasi-Newton methods which are able to solve a vast array of problems (both convex and non-convex). 
* comes with an automatic hyperparameter tuning mechanism thus obviating the need for manually tuning the parameters for any of the included methods. 

## Citation
If you use minSQN for your research, please cite the Github repository:

```
@misc{minSQN2016,
   author = "Nitish Shirish Keskar and Albert S. Berahas",
   title = "{minSQN}: {S}tochastic {Q}uasi-{N}ewton {O}ptimization in {MATLAB}",
   year = "2016",
   url = "https://github.com/keskarnitish/minSQN/",
   note = "[Online]"
 }
```



## Usage Guide

The algorithm can be run using the syntax 

```
logger = minSQN(problem,options,[hyperparameters]);
```
where,
* `problem` is an object pertaining to a specific loss function and data set. 
* `options` is a struct containing the necessary parameters for use in the optimization algorithms. 
* `hyperparameters` is an array of hyperparameters necessary for the optimization algorithms such as the step-size, damping constants and aggregation lengths. This is an optional argument. If it is not specified, minSQN uses its inbuilt automatic tuner (which we describe next) to find hyperparameters. 

### Automatic Tuning:

All method above have certain hyperparameters that need to be set or
tuned. The second column in the table above indicates what
hyperparameters are needed for each of the methods.

In minSQN, we provide an automatic tuning mechanism that randomly samples hyperparameters (as in Bergstra et. al. (2012)) from a prespecified range of hyperparameter values, solves the problem several times, and returns the best optimization run and hyperparameter setting. The number of tuning steps is determined in the options (default is 10).  




### Example (no tuning):
To solve a problem using minSQN, the user must follow 4 steps:

1. Construct the problem class (Logistic Regression and Least Squares are included with the minSQN code. Others can be coded easily using our template)
2. Generate default options using `GenOptions()` and over-write them as needed
3. Set the hyperparameters necessary for the specific method (For instance, SGD requires the step-size, RES requires the step-size and the damping constant and SQN requires the step-size and the aggregation length)
4. Run minSQN


```
X = randn(5000,500);
y = 2*(randn(5000,1) > 0.5) - 1;
problem = lossFunctions.LogReg(X,y);

options = GenOptions();
options.method = 'SQN';
sqn_log_untuned = minSQN(problem,options,[5e-2,5]);
```

The output of `minSQN` would be:

```
sqn_log_untuned = 

              fhist: [21x1 double]
    hyperparameters: [0.050000000000000 5]
             w_star: [500x1 double]
```
where `fhist` is the history of average loss function values over each epoch, `hyperparameters` returns the provided hyperparameters in the case when they are provided and returns their tuned values if the automatic tuning was chosen (see next example). `w_star` is the value of the iterate at the end of the optimization.  

### Example (with tuner):
The process for running the methods with automatic tuning is similar to above except no hyperparameters are passed as input (as in Step 3 above). 
```
X = randn(5000,500);
y = 2*(randn(5000,1) > 0.5) - 1;
problem = lossFunctions.LogReg(X,y);

options = GenOptions();
options.method = 'SQN';
sqn_log_tuned = minSQN(problem,options);
```

The output of `minSQN` in this example would be:
```
sqn_log_tuned = 

              fhist: [21x1 double]
    hyperparameters: [0.006962319523931 26]
             w_star: [500x1 double]
```             
`fhist` and `w_star` are the function values and final iterate as explained in the previous example. In this case, `hyperparameters` is the value of the best hyperparameters as chosen by the automatic tuning mechanism. 

Please refer to `demo.m` for a short demonstration of using `minSQN` for solving a small Logistic Regression problem using three different methods and plotting the results. For a detailed documentation of minSQN and its associated functions, use MATLAB's `help`. For instance, to obtain details about the different options and their significance, use `help GenOptions`.





## References:
SGD:
- Bottou, L., 1998. Online learning and stochastic approximations. On-line learning in neural networks, 17(9), p.142.
 
SQN:
- Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. (2016). A stochastic quasi-Newton method for large-scale optimization. SIAM Journal on Optimization, 26(2), 1008-1031.

oBFGS:
- Schraudolph, N.N., Yu, J. and Günter, S., 2007. A stochastic
quasi-Newton method for online convex optimization. In
International Conference on Artificial Intelligence and Statistics
(pp. 436-443).

oLBFGS:
- Schraudolph, N.N., Yu, J. and Günter, S., 2007. A stochastic
quasi-Newton method for online convex optimization. In
International Conference on Artificial Intelligence and Statistics
(pp. 436-443).
- Mokhtari, A., & Ribeiro, A. (2015). Global convergence of online limited memory bfgs. Journal of Machine Learning Research, 16, 3151-3181.

RES:
- Mokhtari, A. and Ribeiro, A., 2014. Res: Regularized stochastic
bfgs algorithm. Signal Processing, IEEE Transactions on, 62(23),
pp.6089-6104.

SDBFGS:
- Wang, X., Ma, S., Goldfarb, D., & Liu, W. (2014). Stochastic Quasi-Newton Methods for Nonconvex Stochastic Optimization. arXiv preprint arXiv:1607.01231.

adaQN:
- Keskar, N. S., & Berahas, A. S. (2016). adaQN: An Adaptive Quasi-Newton Algorithm for Training RNNs. European Conference Machine Learning and Knowledge Discovery in Databases, (ECML PKDD 2016), Part I, Vol 9851, 1-16.












