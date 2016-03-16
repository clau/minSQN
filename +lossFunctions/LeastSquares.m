classdef LeastSquares < lossFunctions.lossFunction
    % This is a class for carrying out optimization for a Least-Squares
    % fitting problem. Besides the constructor, it has three methods:
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
    % If the indices are not provided, it defaults to using all of the data
    % points. 
        
    properties
        X  % Data matrix
        y  % Target
        m  % Number of data points
        n  % Number of features
        lambda  % Regularization parameter
        w0  % Initial starting point
    end
    methods
        function obj = LeastSquares(X,y)
            % This function takes in the data matrix X and the targets y.
            % The data matrix is in the form [num_of_data_points x
            % num_of_features] and y is with size
            % [num_of_data_points].            

            obj.X = X;
            obj.y = y;
            obj.m = size(X,1);
            obj.n = size(X,2);
            obj.lambda = 1/obj.m;
            obj.w0 = zeros(obj.n,1);%randn(obj.n,1);
        end
        
        function f = funObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.funObj(w,indices) returns the function value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.             
            if(nargin<3)
                indices = 1:obj.m;
            end
            f = 0.5/length(indices)*norm(obj.X(indices,:)*w-obj.y(indices))^2 + obj.lambda/2*norm(w)^2 ;
            if(isnan(f))
                error('Function Value is NaN')
            end
        end
        
        function g = gradObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.gradObj(w,indices) returns the gradient value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.                      
            if(nargin<3)
                indices = 1:obj.m;
            end
            g = 0.5/length(indices)*obj.X(indices,:)'*(obj.X(indices,:)*w-obj.y(indices))+obj.lambda*w;
        end

        function Hv = hessObj(obj,w,v,indices)
            % Given an instantiated object called problem,
            % problem.hessObj(w,v,indices) returns the Hessian-vector product 
            % value at the iterate w, with the vector v, computed on the 
            % data points 'indices'. 
            % If no indices are provided, the function will be computed on
            % all the data points.                         
            
            if(nargin<4)
                indices = 1:obj.m;
            end
            
            Hv = 0.5/length(indices)*obj.X(indices,:)'*(obj.X(indices,:)*v)+obj.lambda*v;
            
        end
        
        
    end
end
