% logger_star = SQN_HvProd(problem,options,hyperparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             STOCHASTIC QUASI-NEWTON (HESSIAN-VECTOR PRODUCT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This subroutine runs a Stochastic Quasi-Newton method on unconstrained
% optimization problem of the form,
%                   min f(x) = sum f_i(x),
% where the 'y' curvature pair is computed using a Hessian-vector product.
%
% Several Stochastic Quasi-Newton methods fall in this category:
%       - SQN               - DSQN
% (see minSQN for further documentation and details about these methods).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%   1) PROBLEM CLASS:
%       - constructs the problem to be solve
%       - creates functions that compute the function value, gradient and
%       Hessian function
%   2) OPTIONS: (see General Options [GenOptions] documentation)
%       - method
%       - epochs
%       - batch size (stochastic gradient)
%       - batch size (Hessian)
%       - LBFGS memory
%       - damping (on/off)
%       - regularization
%       - verbose
%       - tuning_steps [only applies if no steplength and delta has be
%       specified]
%   3) HYPERPARAMETERS:
%       - alpha: constant step length
%       - L: frequency of updating curvature pairs
%       (Note: if no step length and/or regularization parameter is
%       provided the Tuner is used, see minSQN documentation)
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
function logger_star = SQN_HvProd(problem,options,hyperparameters)

% default number of tuning steps
number_of_tuning_steps = 1;
% read the method from the options struct
name_of_method = options.method;

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

% set the Powell damping (true/false) according to the method used
switch name_of_method
    case 'SQN'
        if(options.damping==true)
            warning('SQN does not use Powell Damping. Switching it off. Please use the DSQN method for SQN with Powell Damping.')
            options.damping = false;
        end      
    case 'DSQN'
        if(options.damping==false)
            warning('DSQN uses Powell Damping. Turning it on. Please use the SQN method for DSQN without Powell Damping.')
            options.damping = true;
        end
end

% initialize the best_metric_value (function value) to INF
best_metric_value = Inf;
% for showing progress bar during tuning steps
reverseStr = '';
fprintf('=== Running %s :: === \n',name_of_method);

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
        
        % loop over the epochs
        for epoch=1:options.epochs
            avg_function_value = 0;
            % loop over the batches, batches per epoch =
            % (number of data points)/(batch size + (batch size [Hessian])/L)
            for batch=1:floor(problem.m/(options.batch_size+options.batch_size_hess/L))
                % randomly sample (with replacement) |batch size| indices 
                % from {1,...,n}
                indices = helpers.randperm1(problem.m,options.batch_size);
                % compute a stochastic gradient
                sg = problem.gradObj(w,indices);
                % compute a stochastic function value and add to running
                % sum
                avg_function_value = avg_function_value + problem.funObj(w,indices)/floor(problem.m/(options.batch_size+options.batch_size_hess/L));                
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
                    %
                    if(t>0)
                        %  randomly sample (with replacement)
                        % |batch size [Hessian]| indices from {1,...,n}
                        indices_hess = helpers.randperm1(problem.m,options.batch_size_hess);
                        % update the curvature pairs
                        s = w_n - w_o;
                        y = problem.hessObj(w_n,s,indices_hess);
                        
                        % is Powell damping is true, compute theta
                        if options.damping
                            lhs_sdbfgs = dot(s,y);
                            rhs_sdbfgs = 0.2 * y' * qn.two_loop(y);
                            if(lhs_sdbfgs >= rhs_sdbfgs)
                                theta = 1;
                            else
                                theta = 4 * rhs_sdbfgs / (rhs_sdbfgs/0.2 - lhs_sdbfgs);
                            end
                            r = theta * s + (1-theta)*qn.two_loop(y);
                            qn.store(r,y);
                            
                        else
                            % compute rho and check for sufficient curvature
                            % if curvature not sufficient SKIP the update
                            rho = dot(s,y)/dot(y,y);
                            if(rho>sqrt(eps))
                                % store the curvature pairs
                                qn.store(s,y);
                            else
                                %fprintf('SKIP \n');
                            end
                        end
                    end
                    % update the old averaged uterate
                    w_o = w_n;
                    w_s = zeros(problem.n,1);
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
