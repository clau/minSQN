function [ind] = randperm1(n,k)
% This is just the randperm function as defined in MATLAB. 
% On newer MATLAB versions, the randperm(n,k) returns k unique elements
% from the range 1...n; the older versions of MATLAB don't allow this.
% If you have an older version of MATLAB, the randperm(n,k) doesn't work
% and to use it we do randperm(n) and then pick the first k elements. This
% is potentially very slow. 
% In case you get an error, 
% Comment the line below and uncomment the two lines below it. 

ind = randperm(n,k);

%ind = randperm(n);
%ind = ind(1:k);


end