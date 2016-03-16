classdef QuasiNewton < handle
    % This is a helper class implementing a limited-memory BFGS scheme. 
    % The class has 4 major functions: a constructor, store(), two_loop()
    % and reset. The constructor is used to instantiate a class object once
    % the L-BFGS memory and initialization method (Barzilai-Borwein or
    % ADAGRAD/RMS) has been provided. The store function takes (s,y) pairs
    % and stores them in the class and the two_loop function uses stored
    % (s,y) pairs to compute the inverse-Hessian times vector product. The
    % reset function removes all curvature pairs from memory. 
    % The details for each of the four methods can be found below. 

    properties
        S % Matrix for storing s pairs
        Y % Matrix for storing y pairs
        memory %L-BFGS memory
        initialization_method % Method for initialization H_k^0 at each iteration
        adagrad_sum % Running sum of squared-gradients if ADAGRAD initialization is used
        rms_sum % Running sum of squared-gradients if RMS initialization is used
        
    end
    methods
        function obj = QuasiNewton(memory,initialization_method)
            % This is the QuasiNewton class constructor. As inputs, it
            % takes the L-BFGS memory (usually between 5 and 20) and the
            % initialization method. The initialization method that most
            % SQN methods recommend is the Barzilai-Borwein strategy (Refer
            % to Nocedal and Wright (2006) for more details). We also
            % provide option for using an ADAGRAD- or RMS-like
            % initialization as suggested in adaQN (arxiv.org/abs/1511.01169).
            
            obj.memory = memory;
            obj.initialization_method = initialization_method;
            obj.adagrad_sum = 0;
            obj.rms_sum = 0;
            obj.reset();
        end
        
        function [] = reset(obj)
            % This function resets the S and Y pairs stored.
            obj.S = [];
            obj.Y = [];
        end
        
        function [] = store(obj,s,y)
            % Given an (s,y) pair, this function stores the pair in the
            % object. The code appends the memory pair to the S and Y
            % matrices and removes the first element if the memory has been
            % exceeded. 
            
            obj.S = [obj.S s];
            obj.Y = [obj.Y y];
            
            if(size(obj.S,2)>obj.memory)
                
                obj.S(:,1) = [];
                obj.Y(:,1) = [];
            end
        end
        
        function Hg = two_loop(obj,g)
            % This function carries out the L-BFGS ``two-loop'' recursion
            % for computing the inverse-Hessian-vector product. The input
            % for the function is g (which usually is the gradient of the
            % model). The output is then H_k * g which can be used to take
            % a quasi-Newton step x_{k+1} = x_{k} - \alpha_k * (H_{k} * g_{k}) 
            % Additional details about this recursion can be found in
            % Nocedal and Wright (2006) and also on 
            % Wikipedia (https://en.wikipedia.org/wiki/Limited-memory_BFGS)
            
            % The two-loop recursion requires an initialization of the
            % H_k^0. This is usually done using the Barzilai-Borwein
            % strategy. In adaQN, the authors proposed the use of an
            % ADAGRAD-like strategy for intializing H_k^0 which tends to be
            % non-uniform (in the sense that the initial Hessian is not a
            % multiple of the identity as in the Barzilai-Borwein strategy)
            % and less noisy. 
            
            if(ismember(obj.initialization_method,{'BB','RMS'}))
                obj.adagrad_sum = obj.adagrad_sum + g.^2;
                obj.rms_sum = obj.rms_sum*0.9 + 0.1*g.^2;
            end
            % Initialize Hessian approximation
            
            switch obj.initialization_method
                case 'BB'
                    if(size(obj.S,2)==0)
                        Hg = g;
                        return
                    end
                    
                    gammak = (obj.S(:,end)'*obj.Y(:,end))/(obj.Y(:,end)'*obj.Y(:,end));
                    Hk0 = gammak;
                case 'ADAGRAD'
                    Hk0 = (1./sqrt(obj.adagrad_sum+sqrt(eps)));
                case 'RMS'
                    Hk0 =  (1./sqrt(obj.rms_sum+sqrt(eps)));
                    
                    
            end
            
            % LBFGS two loop recursion
            q = g;
            
            for i = size(obj.S,2):-1:1
                rk(i) = 1/(obj.Y(:,i)'*obj.S(:,i));
                a(i) = rk(i)*obj.S(:,i)'*q;
                q = q - a(i)*obj.Y(:,i);
            end
            
            R = Hk0.*q;
            
            for j = 1:size(obj.S,2)
                beta = rk(j)*obj.Y(:,j)'*R;
                R = R + obj.S(:,j)*(a(j) - beta);
            end
            
            Hg = R;
            
        end
    end
    
end
