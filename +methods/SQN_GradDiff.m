% logger_star = SQN_GradDiff(problem,options,hyperparameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             STOCHASTIC QUASI-NEWTON (GRADIENT DIFFERENCING)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This subroutine runs a Stochastic Quasi-Newton method on unconstrained
% optimization problem of the form,
%                   min f(x) = sum f_i(x),
% where the 'y' curvature pair is computed using gradient differencing.
%
% Several Stochastic Quasi-Newton methods fall in this category:
%       - oBFGS             - oLBFGS
%       - D-oBFGS           - D-oLBFGS
%       - RES               - L-RES
%       - SDBFGS            - L-SDBFGS
% (see minSQN for further documentation and details about these methods).
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%   1) PROBLEM CLASS:
%       - constructs the problem to be solve
%       - creates functions that compute the function value and gradient
%   2) OPTIONS: (see General Options [GenOptions] documentation)
%       - method
%       - epochs
%       - batch size
%       - LBFGS memory
%       - damping (on/off)
%       - regularization
%       - verbose
%       - tuning_steps [only applies if no steplength and delta has be
%       specified]
%   3) HYPERPARAMETERS:
%       - alpha: constant step length
%       - delta: regularizing parameter
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
function logger_star = SQN_GradDiff(problem,options,hyperparameters)

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
    range_for_delta = [1e-5,1e-1];
    warning('Step length (alpha) and damping (delta) has not been provided. Tuning these hyper-parameters for %d steps. The number of tuning steps can be changed using the options struct. Be advised: Tuning takes time.',number_of_tuning_steps)
    % else set the values of the hyperparameters
else
    alpha = hyperparameters(1);
    % if method is one of the following, set delta=0, no Hessian
    % regularization
    if ismember(name_of_method,{'oLBFGS','D-oLBFGS','oBFGS','D-oBFGS'})
        delta = 0;
        
    else
        delta = hyperparameters(2);
    end
    
end

% read the regularization (true, false) from the options struct
use_delta_flag = options.regularization;
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
            delta = use_delta_flag * 10^random('uniform',log10(range_for_delta(1)),log10(range_for_delta(2)));
        end
        
        % initialize the logger struct
        logger.fhist = [];
        
        % set Gamma parameter
        Gamma = 0.1*delta;
        
        % if limited-memory is used, initialize the QN class constructor
        if(options.lbfgs_memory<Inf)
            qn = helpers.QuasiNewton(options.lbfgs_memory,options.H0);
        else
            % initialize BFGS matrix
            B = (delta>0)*delta*speye(problem.n) + (delta==0)*speye(problem.n);
        end
        
        % loop over the epochs
        for epoch=1:options.epochs
            avg_function_value = 0;
            % loop over the batches, batches per epoch = (number of data points)/(2*batch size)
            for batch=1:floor(problem.m/(2*options.batch_size))
                % randomly sample (with replacement) |batch size| indices from {1,...,n}
                indices = helpers.randperm1(problem.m,options.batch_size);
                % compute a stochastic gradient
                sg = problem.gradObj(w,indices);
                
                % compute a stochastic function value and add to running
                % sum
                avg_function_value = avg_function_value + problem.funObj(w,indices)/floor(problem.m/(2*options.batch_size));
                
                % store old iterate
                wo = w;
                % update the iterate, depending on whether the Hessian is
                % regularized in the update or not
                if(options.lbfgs_memory<Inf)
                    if(delta>0)
                        % regularized Hessian and limited memory (LBFGS
                        % updating)
                        [temporary_computation_of_Bg,~,~,~,~] = cgs(@(x) (qn.two_loop(x)+x/delta),sg);
                        w = w - alpha *( sg/delta - 1/delta^2*temporary_computation_of_Bg + Gamma*sg);
                    else
                        % no regularization
                        w = w - alpha * qn.two_loop(sg);
                    end
                else
                    % regularized Hessian and infinite memory (BFGS
                    % updating)
                    w = w - alpha *( B\sg + Gamma*sg);
                end
                % compute a stochastic gradient at the new point, same batch (double gradient evaluations)
                sg_new = problem.gradObj(w,indices);
                % update the curvature pairs
                s = w - wo;
                y = sg_new - sg - delta*s;
                % compute theta (Powell damping)
                if options.damping
                    % finite LBFGS memory
                    if(options.lbfgs_memory<Inf)
                        lhs_sdbfgs = dot(s,y);
                        rhs_sdbfgs = 0.2 * y' * qn.two_loop(y);
                        if(lhs_sdbfgs >= rhs_sdbfgs)
                            theta = 1;
                        else
                            theta = 4 * rhs_sdbfgs / (rhs_sdbfgs/0.2 - lhs_sdbfgs);
                        end
                        r = theta * s + (1-theta)*qn.two_loop(y);
                        % infinite LBFGS memory
                    else
                        lhs_sdbfgs = dot(s,y);
                        rhs_sdbfgs = 0.2 * s' * B * s;
                        if(lhs_sdbfgs >= rhs_sdbfgs)
                            theta = 1;
                        else
                            theta = 4 * rhs_sdbfgs / (rhs_sdbfgs/0.2 - lhs_sdbfgs);
                        end
                        % form r, convex combination of y and Bs
                        r = theta * y + (1-theta)*B*s;
                        
                    end
                else
                    r = s;
                end
                % if limited memory is used, save the curvture pairs
                if options.lbfgs_memory<Inf
                    qn.store(r,y)
                    % else, update the BFGS Hessian approximation
                else
                    B = B + (r*r')/(s'*r) - (B*s*s'*B)/(s'*B*s) + delta*speye(problem.n);
                end
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
            logger_star.hyperparameters = [alpha,delta];
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
