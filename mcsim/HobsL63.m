function [Z] = HobsL63(X,c)

%function to compute the drift in the continuous time observation system 

%INPUTS:
% X = d x N matrix of particles where d = dimension and N = no. of particles 
% c = parameters of obs operator 

Z = NaN*ones(2, size(X,2));
Z(1,:) = c*X(1,:).*X(2,:);
%Z(2,:) = c*X(1,:).*X(3,:);

%slightly modified so that the second component now only observed X_3 (to
%introduce sufficient sign ambiguity)
Z(2,:) = c*X(3,:);

end




