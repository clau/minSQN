% logger_star = adaQN(problem,options,hyperparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   ADAPTIVE QUASI-NEWTON (adaQN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%a
% This subroutine runs the adaQN method on unconstrained optimization
% problem of the form,
%                   min f(x) = sum f_i(x).
% adaQN was originally designed to tackle Deep learning (non-convex)
% problems, more specifically the task of training Recurrent Neural
% Networks (RNNs).
%
% INPUTS:
%   1) PROBLEM CLASS:
%       - constructs the problem to be solve
%       - creates function value, gradient
%   2) OPTIONS: (see General Options [GenOptions] documentation)
%       - method
%       - epochs
%       - batch size (stochastic gradients)
%       - batch size (monitoring set)
%       - LBFGS memory
%       - length of Fisher container
%       - verbose
%       - tuning_steps [only applies if no steplength has be specified]
%   3) HYPERPARAMETERS:
%       - alpha: constant step length
%       (Note: if no step length is provided the Tuner is used, see minSQN
%       documentation)
%       - L: frequency of updating curvature pairs
%       (Note: if no step length and/or regularization parameter is
%       provided the Tuner is used, see minSQN documentation)
%
%
% OUTPUTS:
%   1) LOGGER: for storing information
%       - function value (average function value over last epoch)
%       - optimal parameter vector (w_star)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Nitish Shirish Keskar and Albert S. Berahas
% Date: March 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function logger_star = adaQN(problem,options,hyperparameters)

% default number of tuning steps
number_of_tuning_steps = 1;

% if NO hyperparameters are passed, use the Tuner
if isempty(hyperparameters)
    % specify the the number of tuning steps
    number_of_tuning_steps = options.tuning_steps;
    % specify the ranges of the hyperparameters to be tuned
    range_for_alpha = [1e-6,1e2];
    range_for_L = [2,64];
    warning('Step length (alpha) and averaging length (L) has not been provided. Tuning these hyper-parameters for %d steps. The number of tuning steps can be changed using the options struct. Be advised: Tuning takes time.',number_of_tuning_steps)
    % else set the values of the hyperparameters
else
    alpha = hyperparameters(1);
    L = hyperparameters(2);
end

% initialize the best_metric_value (function value) to INF
best_metric_value = Inf;

% for showing progress bar during tuning steps
reverseStr = '';
fprintf('=== Running adaQN :: === \n');

% loop over the number of tuning steps
for tuning_step = 1:number_of_tuning_steps
    % if function values are NAN, stop tuning step
    try
        % read the initial iterate
        w = problem.w0;
        % if the Tuner is used, randomly sample hyperparameters values in
        % the specified range
        if(number_of_tuning_steps>1)
            percentDone = 100 * tuning_step / number_of_tuning_steps;
            msg = sprintf('Tuning :: Percent done: %3.1f\n', percentDone);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
            alpha = 10^random('uniform',log10(range_for_alpha(1)),log10(range_for_alpha(2)));
            L = randi(range_for_L,1);
        end
        
        % initialize the logger struct
        logger.fhist = [];
        
        % initialize counter, iterations and number of curvature pair
        % updates
        k = 1;
        t = -1;
        % initialize the arrays used for averaging iterates, and storing
        % the old value
        w_s = zeros(problem.n,1);
        w_o = zeros(problem.n,1);
        % initialize the quasi-Newton class
        qn = helpers.QuasiNewton(options.lbfgs_memory,options.H0);
        % create the Fisher container
        fisher_container = [];
        
        % randomly sample the data points labeled 'monitoring set'
        indices_monitor_fun = helpers.randperm1(problem.m,options.batch_size_fun);
        
        % loop over the epochs
        for epoch=1:options.epochs
            % initialize average function value
            avg_function_value = 0;
            
            % loop over the batches, batches per epoch =
            % (number of data points)/(batch size + (batch size [Hessian])/L)
            for batch=1:(floor(problem.m/options.batch_size)-options.batch_size_fun/options.batch_size)
                % randomly sample (with replacement) |batch size| indices
                % from {1,...,n}
                indices = helpers.randperm1(problem.m,options.batch_size);
                % compute a stochastic gradient
                sg = problem.gradObj(w,indices);
                
                % compute a stochastic function value and add to running
                % sum
                avg_function_value = avg_function_value + problem.funObj(w,indices)/(floor(problem.m/options.batch_size)-options.batch_size_fun/options.batch_size);
                
                % append the new stochastic gradient to the Fisher
                % container
                fisher_container = [fisher_container sg];
                % if size of the container exceeds the capacity, delete the
                % oldest stochastic gradient stored
                if(size(fisher_container,2)>options.fisher_memory)
                    fisher_container(:,1) = [];
                end
                % add the current iterate to the running sum over an epoch
                w_s = w_s + w;
                % update the iterate (two-loop recursion to find search
                % direction)
                w = w - alpha * qn.two_loop(sg);
                
                % every L iterations update the curvature pairs
                if(mod(k,L)==0)
                    % increment update counter
                    t = t + 1;
                    % compute the average of the last L iterates
                    w_n = w_s / L;
                    % clear the array containg sum of last L iterates
                    w_s = zeros(problem.n,1);
                    if(t>0)
                        % if new function value is 1.01 times lareger than
                        % the old function value, reset curvature pairs
                        if(problem.funObj(w_n,indices_monitor_fun)>1.01*problem.funObj(w_o,indices_monitor_fun))
                            % reset S and Y
                            qn.reset()
                            w = w_o;
                            continue;
                        end
                        % update the curvature pairs
                        s = w_n - w_o;
                        y = fisher_container * (fisher_container' * s);
                        % if curvature estimate is less than sqrt of
                        % machine precision, SKIP the curvature pair update
                        rho = dot(s,y)/dot(y,y);
                        if(rho>1e-4)
                            qn.store(s,y);
                            w_o = w_n;
                        else
                            %fprintf('SKIP \n');
                        end
                    else
                        % update the old averaged uterate
                        w_o = w_n;
                    end
                    
                    
                end
                % increment iteration counter
                k = k + 1;
            end
            % append the current function value to the logger every epoch
            logger.fhist = [logger.fhist; avg_function_value];
            % if verbose, then print the current function value every epoch
            if(options.verbose)
                fprintf('Epoch: %d, Average Loss: %f \n',epoch,logger.fhist(end))
            end
        end
        
        % if new final function value is better (less) than the current
        % best_metric_value (function value), store the output and
        % hyperparameters for the given run
        if(logger.fhist(end)<best_metric_value)
            logger_star = logger;
            best_metric_value = logger.fhist(end);
            logger_star.hyperparameters = [alpha,L];
            logger_star.w_star = w;
        end
    catch err
        if(strcmp(err.message,'Function Value is NaN'))
            continue
        else
            rethrow(err);
        end
        
    end
    
end
if(~exist('logger_star','var'))
    error('The method diverged. If you are providing your own hyperparameters, please try again. If you are using the automatic tuner, please increase the number of tuning steps using the options struct or chose a different random seed.')
else
    fprintf('Optimization complete :: Reached maximum number of epochs\n')
end
end
