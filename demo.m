% This is a demo for running the files with and without the internal
% automatic tuner. We will run a small Logistic Regression problem using 4
% methods: SGD, oLBFGS and SQN. 
% At the start of each method, we will reset the random number generator
% seed to make sure that we have comparable results. 

%% Let's first start with a blank slate. 
clear;
seed = 0;
rng(seed);
%% Let's now create the Logistic Regression problem using random data

X = sprandn(5000,500,0.01);
y = 2*(randn(5000,1) > 0.5) - 1;

problem = lossFunctions.LogReg(X,y);

%% Let's start off by running SGD with the tuner
% Notice that we don't provide a third argument to minSQN. 
% This automatically tunes the step size.

rng(seed);
options = GenOptions();
options.method = 'SGD';
sgd_log_tuned = minSQN(problem,options);

%% Let's now run SGD *without* the tuner.
% Notice that we provide a third argument to minSQN. 

rng(seed);
options = GenOptions();
options.method = 'SGD';
sgd_log_untuned = minSQN(problem,options,1e-1);

%% Let's now run oLBFGS with the tuner.
% Notice that we don't provide a third argument to minSQN. 

rng(seed);
options = GenOptions();
options.method = 'oLBFGS';
olbfgs_log_tuned = minSQN(problem,options);

%% Let's now run oLBFGS *without* the tuner.
% Notice that we provide a third argument to minSQN. 

rng(seed);
options = GenOptions();
options.method = 'oLBFGS';
olbfgs_log_untuned = minSQN(problem,options,1e-2);

%% Finally, let's do the same for SQN

rng(seed);
options = GenOptions();
options.method = 'SQN';
sqn_log_tuned = minSQN(problem,options);

rng(seed);
options = GenOptions();
options.method = 'SQN';
sqn_log_untuned = minSQN(problem,options,[5e-2,25]);
