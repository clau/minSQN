% This is a demo for running all methods (with automatic tuning) and plot
% the average loss against epochs. 
% Before running/reading this script, we recommend that you look at demo.m
% which is a simpler script that runs a subset of the available methods
% with and without tuning.
% At the start of each method, we will reset the random number generator
% seed to make sure that we have comparable results. 


%% Let's first start with a blank slate. 
clear;
seed = 0;
rng(seed);
%% Let's now create the Logistic Regression problem using `mushroom' dataset
% This dataset was obtained from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

load Data/mushroom.mat

problem = lossFunctions.LogReg(X,y);
problem.w0 = zeros(size(X,2),1);
f0 = problem.funObj(problem.w0); %Only for the purpose of plotting. 

%% Run all methods and plot them.
% method_dict is a cell of some of the available methods. Refer to the
% minSQN help file for more details about them. method_colors is the
% plotting color for the corresponding method. 

method_dict = {'SGD','RES','L-RES','SDBFGS','L-SDBFGS','SQN','DSQN','oBFGS','D-oBFGS','oLBFGS','D-oLBFGS','adaQN'};
method_colors = {'c','k','k--','b','b--','r','r--','g','g--','m','m--','r:'};
for it=1:length(method_dict)
    rng(seed);
    options = GenOptions();
    options.method = method_dict{it};
    logger = minSQN(problem,options);
    semilogy(0:options.epochs,[f0;logger.fhist],method_colors{it},'LineWidth',2);
    hold on;
    drawnow();
end

%% Post-processing
xlabel('Epochs');
ylabel('Average Function Value Per Epoch');
legend(method_dict,'Location','Best');
print -djpeg -r300 all_minSQN_methods.jpeg