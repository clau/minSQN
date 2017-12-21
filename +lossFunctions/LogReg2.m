classdef LogReg2 < lossFunctions.lossFunction
    % This is a class for carrying out optimization for a (Binary) Logistic
    % regression problem. Besides the construction, it has three methods:
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
        Y  % Target
        m  % Number of data points
        c  % Number of classes
        f  % Number of features per class
        n  % Number of total features
        lambda  % Regularization parameter
        w0  % Initial starting point
    end
    methods
        function obj = LogReg2(X,Y)
            % This function takes in the data matrix X and the targets y.
            % The data matrix is in the form [num_of_data_points x
            % num_of_features] and y is either {-1,1} with size
            % [num_of_data_points].
            X = [X ones(size(X, 1), 1)];

            obj.X = X;
            obj.Y = Y;
            obj.c = size(Y,2);  % classes
            obj.f = size(X,2);  % features per class
            obj.m = size(X,1);  % data points
            obj.n = obj.f * obj.c;  % total features
            obj.lambda = 1 / obj.m;
            obj.w0 = zeros(obj.n, 1);%randn(obj.n,1);
        end
        
        % predict
        function p = predictObj(obj,w,indices)
            if(nargin<3)
                indices = 1:obj.m;
            end
            w_ = reshape(w, [obj.f, obj.c]);
            before_softmax = obj.X(indices,:) * w_;
            softmaxed = exp(before_softmax - max(before_softmax, [], 2));
            p = softmaxed ./ sum(softmaxed, 2);
        end

        % loss
        function f = funObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.funObj(w,indices) returns the function value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points. 
            if(nargin<3)
                indices = 1:obj.m;
            end
            pred = obj.predictObj(w,indices);
            loss = -log(pred) .* obj.Y(indices,:);
            f = mean(sum(loss, 2));
            if(isnan(f))
                error('Function Value is NaN')
            end
        end
        
        % d_loss
        function g = gradObj(obj,w,indices)
            % Given an instantiated object called problem,
            % problem.gradObj(w,indices) returns the gradient value at the
            % iterate w computed on the data points 'indices'.
            % If no indices are provided, the function will be computed on
            % all the data points.             
            
            if(nargin<3)
                indices = 1:obj.m;
            end
            pred = obj.predictObj(w,indices);
            delta = pred - obj.Y(indices,:);
            df = zeros(obj.f, obj.c);
            df(1:obj.f-1,:) = obj.X(indices,1:obj.f-1)' * delta / length(indices);
            df(obj.f,:) = mean(delta, 1);
            g = reshape(df, [obj.n, 1]);
        end
        
        % hess
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
            
            error('Not implemented')
            temp_save = exp(-1*(obj.y(indices)).*(obj.X(indices,:)*w));
            b = temp_save ./ (1+temp_save);
            Hv = 1/length(indices)*obj.X(indices,:)'*(diag(b-b.^2)*(obj.X(indices,:)*v))+obj.lambda*v;
            
        end
        
        
    end
end
