% options = GenOptions();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         GENERAL OPTIONS for minSQN 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function generates all the options necessary to run the minSQN
% function.
%
% Note: not all options are needed for every method.
%
% OPTIONS:
%   - method: determines which SQN method to use
%       * default value: options.method = 'SQN'; 
%       * recommended range of values: {'SGD','RES','L-RES','SDBFGS','L-SDBFGS','SQN','DSQN','oBFGS','D-oBFGS','oLBFGS','D-oLBFGS','adaQN'}
%
%   - epochs: number of epochs (passes through the dataset) to run the 
%   method
%       * default value: options.epochs = 20;
%       * recommended range of values: Any positive integer
%
%   - batch size: number of data points used in the gradient evaluation
%       * default value: options.batch_size = 256;
%       * recommended range of values: 32 - 1024
%       Larger batches occupy more memory and might process faster but
%       given a fixed epoch budget, might lead to inferior results.
%
%   - batch size (Hessian): number of data points used in the Hessian
%   evaluation [only applies to methods that compute the  y curvature pair 
%   using Hessian-vector products]
%       * default value: options.batch_size_hess = 10*options.batch_size;
%       * recommended range of values: [2 to 20] * options.batch_size
%
%   - batch size (Function): number of data points used in the function
%   evaluation (once an epoch) [only applies to adaQN, monitoring set]
%       * default value: options.batch_size_fun = options.batch_size;
%       * recommended range of values: [1 to 10] * options.batch_size 
%
%   - LBFGS memory: number of curvature pairs to store in memory to be used
%   in the inverse Hessian matrix approximation [only applies to method 
%   limited-memory variants of the algorithms]
%       * default value: options.lbfgs_memory = 20;
%       * recommended range of values: 5-20
%
%   - fisher memory: maximum length of Fisher matrix container [only 
%   applies to adaQN]
%       * default value: options.fisher_memory = 100;
%       * recommended range of values: 50-200
%
%   - verbose: level of printing (verbose = 0, no printing; verbose = 1,
%   print function value every iteration)
%       * default value: options.verbose = 0;
%       * recommended range of values: 0 or 1
%
%   - tuning_steps: number of different sets of hyperparameters to test
%   [only applies if the hyperparameters are not set
%       * default value: options.tuning_steps = 10;
%       * recommended range of values: 10-100. 
%       More tuning steps might improve hyperparameters chosen but will
%       increase time needed for result.
%
%   - damping: Powell damping ((Nocedal and Wright, 2006, p.537)) used in curvature pair updates
%   (false, no Powell damping; true, Powell damping)
%       * default value: options.damping = true;
%       * recommended range of values: true or false
%
%   - regularization: regularization of the inverse Hessian approximation
%   (false, delta=0; true delta>0)
%       * default value: options.regularization = false;
%       * recommended range of values: true or false
%
%   - H0: initial Hessian approximation [only applies to limited-memory 
%   variants of the algorithm]
%       * default value: options.H0 = 'BB';
%       * recommended range of values: {'BB','ADAGRAD','RMS'}
%
%   - method_for_computing_y: gradient differencing vs. Hessian-vector
%   product
%       * no default value
%       * recommended range of values: 'GradDiff' or 'HvProd'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Nitish Shirish Keskar and Albert S. Berahas
% Date: March 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function options = GenOptions()
options.method = 'SQN';
options.epochs = 20;
options.batch_size = 256;
options.batch_size_hess = 10*options.batch_size;
options.batch_size_fun = options.batch_size;
options.lbfgs_memory = 20;
options.fisher_memory = 100;
options.verbose=0;
options.tuning_steps = 10;
options.damping = true;
options.regularization = true;
options.H0 = 'BB';
%options.method_for_computing_y = [];
end