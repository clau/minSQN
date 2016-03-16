% logger_star = SGD(problem,options,alpha);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   STOCHASTIC GRADIENT DESCENT (SGD)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This subroutine runs the SGD method on unconstrained optimization problem
% of the form,
%                   min f(x) = sum f_i(x).
%
%
% INPUTS:
%   1) PROBLEM CLASS:
%       - constructs the problem to be solve
%       - creates function value, gradient
%   2) OPTIONS: (see General Options [GenOptions] documentation)
%       - method
%       - epochs
%       - batch size
%       - verbose
%       - tuning_steps [only applies if no steplength has be specified]
%   3) HYPERPARAMETERS:
%       - alpha: constant step length
%       (Note: if no step length is provided the Tuner is used, see minSQN
%       documentation)
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
function logger_star = SGD(problem,options,hyperparameters)

% default number of tuning steps
number_of_tuning_steps = 1;

% if NO hyperparameter are passed, use the Tuner
if(isempty(hyperparameters))
    % specify the the number of tuning steps
    number_of_tuning_steps = options.tuning_steps;
    % specify the range of the hyperparameter to be tuned
    range_for_alpha = [1e-6,1e2];
    warning('Step length (alpha) has not been provided. Tuning alpha for %d steps. The number of tuning steps can be changed using the options struct. Be advised: Tuning takes time.',number_of_tuning_steps)
else
    alpha = hyperparameters(1);
end


% initialize the best_metric_value (function value) to INF
best_metric_value = Inf;

% for showing progress bar during tuning steps
reverseStr = '';


fprintf('=== Running SGD :: === \n');
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
        end
        
        % initialize the logger struct
        logger.fhist = [];
        
        % loop over the epochs
        for epoch=1:options.epochs
            avg_function_value = 0;
            
            % loop over the batches, batches per epoch =
            % (number of data points)/(batch size)
            for batch=1:floor(problem.m/options.batch_size)
                % randomly sample (with replacement) |batch size| indices from
                %{1,...,n}
                indices = helpers.randperm1(problem.m,options.batch_size);
                % compute a stochastic gradient
                sg = problem.gradObj(w,indices);
                
                % compute a stochastic function value and add to running
                % sum
                avg_function_value = avg_function_value + problem.funObj(w,indices)/floor(problem.m/options.batch_size);
                
                % update the iterate
                w = w - alpha * sg;
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
            logger_star.alpha = alpha;
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