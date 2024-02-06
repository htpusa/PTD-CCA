function [optC res] = tunePTDCCA(X,varargin)

% tunePTDCCA
% [optC res] = tunePTDCCA(X) finds the optimal regularisation parameter c
%                   for PTDCCA
% [optC res] = tunePTDCCA(X,'mode',"view") optimises the parameter c for
%                   each view
% [optC res] = tunePTDCCA(X,'D',3) evaluates performance for the first 3
%                   canonical tuples (by default only the first is
%                   considered)
%
%   tunePTDCCA uses model consistency to find the best hyperparameter 
%   value(s): data is split into two and PTDCCA is run for both folds. 
%   The best model is the one where the angle between the coefficient 
%   vectors from the two folds is the smallest.
%   Note that "view"-mode uses the PTDCCApath function to generate
%   models and so "cross-models" where c is different for different views
%   are likely suboptimal. However, this should be MUCH faster than an
%   exhaustive grid search.
%
%   INPUTS:
%   X           -   Mx1 cell where M is the number of views and X{M} is a
%                       NxPm matrix with N rows corresponding to samples and
%                       Pm columnds to variables
%   OPTIONAL INPUTS:
%   'mode'      -   "global" or "view", whether to find a single global
%                       parameter or one for each view (default: "global")
%   'D'         -   double, the number of canonical variable tuples
%                       (default: 1)
%   'L'         -   double, length of the sequence ie how many different
%                       c values are used (default: 20)
%   'c'         -   vector, the sequence of c values to use
%                       should be in ascending order
%                       (default: linspace(0,1,L))
%   'initType'  -   "tensor" or "matrix, how to initialise the algorithm
%                       (default: "tensor")
%                       "tensor" uses the singular vectors of the
%                       cross-covariance tensor. Probably the best option but
%                       costly for larger data
%                       "matrix" uses the singular vectors of the pairwise
%                       cross-covariance matrices
%   'maxIter'   -   maximum number of iterations (default: 1000)
%   'eps'       -   stopping criterion threshold (default: 1e-10)
%   'rounds'    -   how many rounds of CV are done (default: 10)
%
%   OUTPUTS:
%   'optC'      -   optimal c value if 'mode' is "global"
%                   or
%                   1xM vector, optimal c value for each view if 'mode' is
%                   "view"
%   'res'       -   res.score contains the average coefficient vector
%                   angles for each c value tested
%                   res.c contains the c values tested
%                   res.ind constains the index or indices of the optimal c
%                   value(s)
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG Cylinders Model_Year];
%      nans = sum(isnan(data),2) > 0;
%      X = {data(~nans,1:2); data(~nans,3:4); data(~nans,5:end)};
%      optC = tunePTDCCA(X);

%   Author: T.Pusa, 2024

% default parameters
mode = "global";
L = 20;
D = 1;
c = [];
param.maxIter = 1000;
param.eps = 1e-10;
init = "tensor";
rounds = 10;

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'mode'
					mode = varargin{1, i+1};
                case 'D'
					D = varargin{1, i+1};
                case 'c'
					c = varargin{1, i+1};
                case 'L'
					L = varargin{1, i+1};
                case 'maxIter'
					param.maxIter = varargin{1, i+1};
                case 'eps'
					param.eps = varargin{1, i+1};
                case 'initType'
					init = varargin{1, i+1};
                case 'rounds'
					rounds = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

M = numel(X);
N = size(X{1},1);
if ~isempty(c)
    L = numel(c);
end

if mode=="global"
    score = zeros(L,rounds);
elseif mode=="view"
    score = zeros(L^M,rounds);
else
    error('No such mode')
end

for r=1:rounds
    cv = crossvalind('KFold',N,2);
    X1 = cellfun(@(Xm) Xm(cv~=1,:),X,'UniformOutput',false);
    X2 = cellfun(@(Xm) Xm(cv==1,:),X,'UniformOutput',false);
    [W1,~,~,c] = PTDCCApath(X1,'c',c,'L',L,'D',D,'initType',init);
    W2 = PTDCCApath(X2,'c',c,'L',L,'D',D,'initType',init);
    for i=1:size(score,1)
        if mode=="global"
            inds = i*ones(M,1);
        else
            inds = cell(1,M);
            [inds{:}] = ind2sub(L*ones(1,M),i);
            inds = cell2mat(inds);
        end
        ctmp = zeros(M,1);
        for m=1:M
            ctmp(m) = c(inds(m));
        end
        tmp = 0;
        for d=1:D
            w1 = arrayfun(@(m) W1{m}(:,inds(m),d),1:M,'UniformOutput',false)';
            w2 = arrayfun(@(m) W2{m}(:,inds(m),d),1:M,'UniformOutput',false)';
            tmp = tmp + mean(cellfun(@(a,b) acos((a'*b)/(norm(a)*norm(b))),w1,w2));
        end
        score(i,r) = tmp/D;
    end
end
score = mean(score,2);
[~,I]=min(score);
if mode=="global"
    optC = c(I);
    res.score = score;
    res.ind = I;
else
    inds = cell(1,M);
    [inds{:}] = ind2sub(L*ones(1,M),I);
    optC = zeros(1,M);
    for m=1:M
        optC(m) = c(inds{m});
    end
    res.score = reshape(score,L*ones(1,M));
    res.ind = cell2mat(inds);
end
res.c = c;