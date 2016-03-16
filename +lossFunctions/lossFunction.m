classdef (Abstract) lossFunction < handle
    % This is a template class for carrying out optimization for a loss
    % function. Besides the construction (which sets the data matrix and 
    % targets), it has three methods:
    % funObj (which computes the function value), gradObj (which computes
    % the gradient value) and hessObj (which computes a Hessian-vector
    % product). The funObj and gradObj take as inputs the current iterate
    % and the indices on which the function/gradient is to be computed. The
    % hessObj also takes as input the vector which is to be multiplied to
    % the Hessian. 
    % In other words, for an iterate w, for the first 10 data points, the
    % function, gradient and Hessian-vector product (with the vector of
    % ones) can be computed as
    % funObj(w,1:10), gradObj(w,1:10), hessObj(w,ones(size(w),1:10)
    % respectively.
    % If the indices are not provided, it must default to using all of the data
    % points. 
        
    methods (Abstract)

            funObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.funObj(w,indices) returns the function value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.             
        
            gradObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.gradObj(w,indices) returns the gradient value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.                      

            hessObj(obj,w,v,indices)
            % Given an instantiated object called problem,
            % problem.hessObj(w,v,indices) returns the Hessian-vector product 
            % value at the iterate w, with the vector v, computed on the 
            % data points 'indices'. 
            % If no indices are provided, the function will be computed on
            % all the data points.                         
           

        
       
    end
end
