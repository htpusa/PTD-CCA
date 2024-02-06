function C = crosscovten(X)

% CROSSCOVTEN cross-covariance tensor
%   C = crosscovten(X) calculates the cross-covariance tensor C for data 
%   matrices stored in cell vector X.
%
%   INPUTS:
%   X       -   M-by-1 cell vector where X{m} is an N-by-pm data matrix
%
%   OUTPUTS:
%   C       -   p1-by-p2-by-...-by-pm matrix where the entry C(i,j,k) is
%               the "higher order" covariance between X{1}(:,i), X{2}(:,j),
%               and X{3}(:,k)

M = numel(X);
D = cellfun(@(x) size(x,2),X);
N = size(X{1},1);
C = zeros(D');
for n=1:N
    nC = X{1}(n,:);
    for m=2:M
        nC = tensorprod(nC,X{m}(n,:));
    end
    C = C+squeeze(nC);
end