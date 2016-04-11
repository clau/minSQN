% log_struct = minSQN(problem,options,hyperparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                minSQN 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% # minSQN : Stochastic Quasi-Newton Optimization in MATLAB
% 
% Authors: [Nitish Shirish Keskar](http://users.iems.northwestern.edu/~nitish/) 
% and [Albert S. Berahas](https://sites.google.com/a/u.northwestern.edu/albertsberahas/home)
% 
% ## Introduction
% 
% This is a package for solving an unconstrained minimization
% problem of the form,
% min f(x) = sum_i f_i(x).
% 
% minSQN allows for the user to solve large-scale (sum-of-functions)
% optimization problems using one of 11 Stochastic Quasi-Newton methods.
% 
% The following table summarizes all the methods that minSQN contains. The
% methods are classified in terms of:
% - Hyperparameters
% - Length of LBFGS memory (limited/inf) [if inf, BFGS method used]
% - Powell damping
% - Hessian damping
% - Curvature pair update (y)
% 
% ```
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |  Method  | Hyperparameters | LBFGS Memory | Powell Damping | Hessian Damping |     Curvature pair     |          Reference          |
% |          |                 | (finite/inf) |      (Y/N)     |      (Y/N)      |       (y) update       |                             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |    SQN   |     alpha, L    |    finite    |        N       |        N        | Hessian-vector product |     Byrd et. al. (2014)     |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |   DSQN   |     alpha, L    |    finite    |        Y       |        N        | Hessian-vector product |              --             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |   oBFGS  |   alpha, delta  |      inf     |        N       |        Y        |  Gradient differencing |  Schraudolph et. al. (2007) |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |  oLBFGS  |   alpha, delta  |    finite    |        N       |        N        |  Gradient differencing | Schraudolph et. al. (2007), |
% |          |                 |              |                |                 |                        |   Mokhtari et. al. (2014)   |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |  D-oBFGS |   alpha, delta  |      inf     |        Y       |        Y        |  Gradient differencing |              --             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% | D-oLBFGS |   alpha, delta  |    finite    |        Y       |        N        |  Gradient differencing |              --             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |    RES   |   alpha, delta  |      inf     |        N       |        Y        |  Gradient differencing |   Mokhtari et. al. (2014)   |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |   L-RES  |   alpha, delta  |    finite    |        N       |        N        |  Gradient differencing |                             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |  SDBFGS  |   alpha, delta  |      inf     |        Y       |        Y        |  Gradient differencing |     Wang et. al. (2014)     |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% | L-SDBFGS |   alpha, delta  |    finite    |        Y       |        Y        |  Gradient differencing |              --             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% |   adaQN  |     alpha, L    |    finite    |        N       |        N        | Hessian-vector product |    Keskar et. al. (2015)    |
% |          |                 |              |                |                 |   accumulated Fisher   |                             |
% +----------+-----------------+--------------+----------------+-----------------+------------------------+-----------------------------+
% ```
%
% ### Features
% The minSQN package
% 
% * is written in pure-MATLAB with minimal dependencies and emphasizes 
% simplicity, extendibility and cross-platform compatibility. 
% * allows the user to run 11 different stochastic quasi-Newton methods 
% which are able to solve a vast array of problems (both convex and 
% non-convex). 
% * comes with an automatic hyperparameter tuning mechanism thus obviating 
% the need for manually tuning the parameters for any of the included 
% methods. 
% 
% ## Citation
% If you use minSQN for your research, please cite the Github repository:
% 
% ```
% @misc{minSQN2016,
%    author = "Nitish Shirish Keskar and Albert S. Berahas",
%    title = "{minSQN}: {S}tochastic {Q}uasi-{N}ewton {O}ptimization in {MATLAB}",
%    year = "2016",
%    url = "https://github.com/keskarnitish/minSQN/",
%    note = "[Online]"
%  }
% ```
%
% ## Usage Guide
% 
% The algorithm can be run using the syntax 
% 
% ```
% logger = minSQN(problem,options,[hyperparameters]);
% ```
% where,
% * `problem` is an object pertaining to a specific loss function and data 
% set. 
% * `options` is a struct containing the necessary parameters for use in 
% the optimization algorithms. 
% * `hyperparameters` is an array of hyperparameters necessary for the 
% optimization algorithms such as the step-size, damping constants and 
% aggregation lengths. This is an optional argument. If it is not 
% specified, minSQN uses its inbuilt automatic tuner (which we describe 
% next) to find hyperparameters. 
% 
% ### Automatic Tuning:
% 
% All method above have certain hyperparameters that need to be set or
% tuned. The second column in the table above indicates what
% hyperparameters are needed for each of the methods.
% 
% In minSQN, we provide an automatic tuning mechanism that randomly samples 
% hyperparameters (as in Bergstra et. al. (2012)) from a prespecified range 
% of hyperparameter values, solves the problem several times, and returns 
% the best optimization run and hyperparameter setting. The number of 
% tuning steps is determined in the options (default is 10).  
% 
% ### Example (no tuning):
% To solve a problem using minSQN, the user must follow 4 steps:
% 
% 1. Construct the problem class (Logistic Regression and Least Squares are
% included with the minSQN code. Others can be coded easily using our 
% template)
% 2. Generate default options using `GenOptions()` and over-write them as 
% needed
% 3. Set the hyperparameters necessary for the specific method (For 
% instance, SGD requires the step-size, RES requires the step-size and the 
% damping constant and SQN requires the step-size and the aggregation 
% length)
% 4. Run minSQN
%  
% ```
% X = randn(2000,500);
% y = 2*(randn(2000,1) > 0.5) - 1;
% problem = lossFunctions.LogReg(X,y);
% 
% options = GenOptions();
% options.method = 'SQN';
% sqn_log_untuned = minSQN(problem,options,[5e-2,5]);
% ```
% 
% The output of `minSQN` would be:
% 
% ```
% sqn_log_untuned = 
% 
%               fhist: [21x1 double]
%     hyperparameters: [0.050000000000000 5]
%              w_star: [500x1 double]
% ```
% where `fhist` is the history of average loss function values over each 
% epoch, `hyperparameters` returns the provided hyperparameters in the case
% when they are provided and returns their tuned values if the automatic 
% tuning was chosen (see next example). `w_star` is the value of the 
% iterate at the end of the optimization.  
% 
% ### Example (with tuner):
% The process for running the methods with automatic tuning is similar to 
% above except no hyperparameters are passed as input (as in Step 3 above). 
% ```
% X = randn(2000,500);
% y = 2*(randn(2000,1) > 0.5) - 1;
% problem = lossFunctions.LogReg(X,y);
% 
% options = GenOptions();
% options.method = 'SQN';
% sqn_log_tuned = minSQN(problem,options);
% ```
% 
% The output of `minSQN` in this example would be:
% ```
% sqn_log_tuned = 
% 
%               fhist: [21x1 double]
%     hyperparameters: [0.006962319523931 26]
%              w_star: [500x1 double]
% ```             
% `fhist` and `w_star` are the function values and final iterate as 
% explained in the previous example. In this case, `hyperparameters` is the
% value of the best hyperparameters as chosen by the automatic tuning 
% mechanism. 
% 
% Please refer to `demo.m` for a short demonstration of using `minSQN` for 
% solving a small Logistic Regression problem using three different methods
% and plotting the results. For a detailed documentation of minSQN and its 
% associated functions, use MATLAB's `help`. For instance, to obtain 
% details about the different options and their significance, use `help 
% GenOptions`.
% 
% 
% ## References:
% SGD:
% - Bottou, L., 1998. Online learning and stochastic approximations. 
% On-line learning in neural networks, 17(9), p.142.
%  
% SQN:
% - Byrd, R.H., Hansen, S.L., Nocedal, J. and Singer, Y., 2014.
% A stochastic quasi-Newton method for large-scale optimization.
% arXiv preprint arXiv:1401.7020.
% 
% oBFGS:
% - Schraudolph, N.N., Yu, J. and Günter, S., 2007. A stochastic
% quasi-Newton method for online convex optimization. In
% International Conference on Artificial Intelligence and Statistics
% (pp. 436-443).
% 
% oLBFGS:
% - Schraudolph, N.N., Yu, J. and Günter, S., 2007. A stochastic
% quasi-Newton method for online convex optimization. In
% International Conference on Artificial Intelligence and Statistics
% (pp. 436-443).
% - Mokhtari, A. and Ribeiro, A., 2014. Global convergence of online
% limited memory bfgs. arXiv preprint arXiv:1409.2045.
% 
% RES:
% - Mokhtari, A. and Ribeiro, A., 2014. Res: Regularized stochastic
% bfgs algorithm. Signal Processing, IEEE Transactions on, 62(23),
% pp.6089-6104.
% 
% SDBFGS:
% - Wang, X., Ma, S. and Liu, W., 2014. Stochastic Quasi-Newton
% Methods for Nonconvex Stochastic Optimization. arXiv preprint
% arXiv:1412.1196.
% 
% adaQN:
% - Keskar, N.S. and Berahas, A.S., 2015. adaQN: An Adaptive
% Quasi-Newton Algorithm for Training RNNs. arXiv preprint
% arXiv:1511.01169.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Disclaimer: the optimization methods contained in minSQN were not
% designed by the authors of this code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Nitish Shirish Keskar and Albert S. Berahas
% Date: March 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The MIT License (MIT)
% 
% Copyright (c) 2016 Nitish Shirish Keskar and Albert S. Berahas
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function log_struct = minSQN(problem,options,hyperparameters)

% if the number of input arguments is:
%       0, Error no Problem and options input
%       1, Error no options input
%       2, set the hyperparameters to empty (use tuner)
switch nargin
    case 0
        error('Problem and options not specified. Please construct a problem class and use use GenOptions() to define default options.')
    case 1
        error('Options not specified. Please use GenOptions() to define default options.')
    case 2
        hyperparameters = [];
    case 3
        %Do Nothing
    otherwise
        error('Not Implemented.')
end


% Check the length of LBFGS memory to be used:
%   if method requires infinite length memory (BFGS updating), set the 
%       length to infinite
%   else if the method requires finite length memory (LBFGS updating) and 
%       memory length is set to infinite, set the length to the default
%       option (20)
%   else proceed with finite length memory (LBFGS updating)
if ismember(options.method,{'RES','SDBFGS','oBFGS','D-oBFGS'})
    options.lbfgs_memory = Inf;
else
    if(~(options.lbfgs_memory<Inf))
        warning('You have requested a limited-memory method but have set options.lbfgs_memory to Infinity. Setting L-BFGS memory to 20 instead.\n');
        options.lbfgs_memory = 20;
    end
end

% choose the method
switch options.method
    case {'SGD'}
        log_struct = methods.SGD(problem,options,hyperparameters);
    % RES and L-RES: regularization on, damping off
    case {'RES','L-RES'}
        options.regularization = true;
        options.damping = false;
        log_struct = methods.SQN_GradDiff(problem,options,hyperparameters);
    % oLBFGS and oBFGS: regularization off, damping off
    case {'oLBFGS','oBFGS'}
        options.regularization = false;
        options.damping = false;
        log_struct = methods.SQN_GradDiff(problem,options,hyperparameters);
    % D-oLBFGS and D-oBFGS: regularization off, damping on
    case {'D-oLBFGS','D-oBFGS'}
        options.regularization = false;
        options.damping = true;
        log_struct = methods.SQN_GradDiff(problem,options,hyperparameters);
    % SDBFGS and L-SDBFGS: regularization on, damping on
    case {'SDBFGS','L-SDBFGS'}
        options.regularization = true;
        options.damping = true;
        log_struct = methods.SQN_GradDiff(problem,options,hyperparameters);
    % SQN: regularization off, damping off;
    case {'SQN'}
        options.damping = false;
        log_struct = methods.SQN_HvProd(problem,options,hyperparameters);
    % DSQN: regularization off, damping on    
    case {'DSQN'}
        options.damping = true;
        log_struct = methods.SQN_HvProd(problem,options,hyperparameters);
    % adaQN
    case {'adaQN'}
        if(~ (strcmp(options.H0,'ADAGRAD') || strcmp(options.H0,'RMS')))
            options.H0 = 'RMS';
        end
        log_struct = methods.adaQN(problem,options,hyperparameters);
    % safeguarding........
    otherwise
        if(exist('options.method_for_computing_y'))
            switch options.method_for_computing_y
                case 'GradDiff'
                    log_struct = methods.SQN_GradDiff(problem,options,hyperparameters);
                case 'HvProd'
                    log_struct = methods.SQN_HvProd(problem,options,hyperparameters);
                otherwise
                    error('This option for computing y has not been implemented.')
            end
        else
            error('If you wish to use an algorithm not in our dictionary, you must specific how the computation of y is done ("GradDiff" or "HvProd"). Refer to the help file of GenOptions for more details.')
        end
end
end

