% This is a demo for running all methods (with automatic tuning) and plot
% the average loss against epochs. 
% Before running/reading this script, we recommend that you look at demo.m
% which is a simpler script that runs a subset of the available methods
% with and without tuning.
% At the start of each method, we will reset the random number generator
% seed to make sure that we have comparable results. 


%% Let's first start with a blank slate. 
% clear;
seed = 0;
rng(seed);
%% Let's now create the Logistic Regression problem using `mushroom' dataset
% This dataset was obtained from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html

% X = dlmread('Data/mnist/train_data.csv',' ');
% Z = dlmread('Data/mnist/train_label.csv',' ');
% VX = dlmread('Data/mnist/val_data.csv',' ');
% VZ = dlmread('Data/mnist/val_label.csv',' ');

% samples = 5000;
% [xs, idx] = datasample(X, samples, 'Replace', false);
% zs = Z(idx,:);

% problem = lossFunctions.LogReg2(X,Z);
problem = lossFunctions.LogReg2(xs,zs);
problem.w0 = zeros(problem.n, 1);
f0 = problem.funObj(problem.w0); %Only for the purpose of plotting. 

%% Run all methods and plot them.
% method_dict is a cell of some of the available methods. Refer to the
% minSQN help file for more details about them. method_colors is the
% plotting color for the corresponding method. 
% 'RES','SDBFGS','oBFGS','D-oBFGS'

method_dict = {'SGD','oLBFGS','adaQN'};
method_colors = {'c','b','r:'};


% tic
% my_program
% wtime = toc
% fprintf ( 1, '  MY_PROGRAM took %f seconds to run.\n', wtime );

% method_dict = {'SGD','RES','L-RES','SDBFGS','L-SDBFGS','oBFGS','D-oBFGS','oLBFGS','D-oLBFGS','adaQN'};
% method_colors = {'c','k','k--','b','b--','g','g--','m','m--','r:'};
% method_dict = {'adaQN'};
% method_colors = {'r:'};

for it=1:length(method_dict)
    rng(seed);
    options = GenOptions();
    options.method = method_dict{it};
    logger = minSQN(problem,options);
    logger.fhist
    mean(logger.durations)
    semilogy(0:options.epochs,[f0;logger.fhist],method_colors{it},'LineWidth',2);
    hold on;
    drawnow();
end

%% Post-processing
xlabel('Epochs');
ylabel('Average Function Value Per Epoch');
legend(method_dict,'Location','Best');
print -djpeg -r300 all_minSQN_methods2.jpeg