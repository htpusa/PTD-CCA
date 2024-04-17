function [W r V] = PTDCCA(X,c,varargin)

% PTDCCA Sparse tensor CCA via penalised tensor decomposition
%   [W r V] = PTD(X,c) performs sparse canonical correlation analysis for
%   two or more data matrices in cell vector X. The algorithm is 
%   an extension of the penalised matrix decomposition CCA presented in 
%   Witten et al. (2009).
%   [W r V] = PTD(X,c,'D',3) calculates the first three canonical vector
%   tuples.
%
%   INPUTS:
%   X           -   Mx1 cell where M is the number of views and X{m} is a
%                       NxPm matrix with N rows corresponding to samples and
%                       Pm columns to variables
%   c           -   double in the interval [0,1], global sparsity parameter
%                   or
%                   length-M vector where c(m) is the sparsity parameter for
%                       view m
%                   c=0 produces a maximally sparse model while at c=1 no
%                       sparsity is enforced
%   OPTIONAL INPUTS:
%   'D'         -   double, the number of canonical variable tuples
%                       (default: 1)
%   'initType'  -   "tensor" or "random", how to initialise the algorithm
%                       (default: "tensor")
%                       "tensor" uses the singular vectors of the
%                       cross-covariance tensor. Probably the best option but
%                       costly for larger data
%                       "random" tries several random starts
%   'CCten'     -   P1xP2x...xPm tensor, the cross-variance tensor of the data
%                       in X (C = crosscovten(X))
%                       If the algorithm is run multiple times, time can be
%                       saved by calculating the tensor offline
%   'rStarts'   -   how many random starts are tried (default: 5)
%   'maxIter'   -   maximum number of iterations (default: 1000)
%   'eps'       -   stopping criterion threshold (default: 1e-10)
%
%   OUTPUTS:
%   'W'         -   Mx1 cell where W{m} is an PmxD matrix where each column
%                       is a vector of canonical coefficients for view m
%   'r'         -   Dx1 vector of objective values
%   'V'         -   Mx1 cell where V{m} is a NxD matrix where each column
%                       is vector of canonical variables for view m
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG Cylinders Model_Year];
%      nans = sum(isnan(data),2) > 0;
%      X = {data(~nans,1:2); data(~nans,3:4); data(~nans,5:end)};
%      [W r V] = PTD(X,0.5);

%   References:
%       Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. 
%       "A penalized matrix decomposition, with applications to sparse 
%       principal components and canonical correlation analysis." 
%       Biostatistics 10.3 (2009): 515-534.

%   Author: T.Pusa, 2024

% default parameters
D = 1;
param.maxIter = 1000;
param.eps = 1e-10;
CCten = [];
init = "tensor";
coolDown = 1;
rInits = 5;

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
                case 'D'
					D = varargin{1, i+1};
                case 'maxIter'
					param.maxIter = varargin{1, i+1};
                case 'eps'
					param.eps = varargin{1, i+1};
                case 'rStarts'
					rInits = varargin{1, i+1};
                case 'CCten'
					CCten = varargin{1, i+1};
                case 'initType'
					init = varargin{1, i+1};
                otherwise
					error(['Could not recognise optional input names.' ...
                        '\nNo input named "%s"'],...
						varargin{1,i});
            end
        end
    end
end

M = numel(X);
if ~iscell(X)
    error('X should be a cell')
elseif size(X,1)~=M
    error('Views in X should be in rows')
elseif M<2
    error('There should be at least 2 views')
end

p = cellfun(@(x) size(x,2),X);
N = size(X{1},1);

if isempty(CCten)
    if init=="tensor"
        try
            CCten = crosscovten(X);
            wInits = lathauwerSVDs(CCten,D);
        catch
            warning(['Failed to calculate cross-covariance tensor, ' ...
                'defaulting to random initialisation.'])
            init = "random";
        end
    end
else
    if any(size(CCten)'~=p)
        error('Dimensions of cross-covariance tensor do not match X')
    end
end

for m=2:M
    if size(X{m},1)~=N
        error('All views should have the same number of samples')
    end
end

if any(c<0) | any(c>1)
    error('c should be between 0 and 1')
end

if (M~=numel(c)) & (numel(c)~=1)
    error('c should be a double or have a value for each view')
else
    c = c.*(sqrt(p')-1)+1;
end

W = arrayfun(@(pm) zeros(pm,D),p,'UniformOutput',false);
r = zeros(1,D);
V = arrayfun(@(n) zeros(n,D),N*ones(M,1),'UniformOutput',false);

X = cellfun(@(Xm) Xm - mean(Xm,1),X,'UniformOutput',false);

for d=1:D
    if init=="random"
        for rI=1:rInits
            wInit = arrayfun(@(m) rand(p(m),D)-0.5,1:M,'UniformOutput',0)';
            if coolDown
                wInit = PTDfromInit(X,sqrt(p'),wInit,param);
            end
            [wtmp,rtmp] = PTDfromInit(X,c,wInit,param);
            if rtmp>r(d)
                w = wtmp;
                r(d) = rtmp;
            end
        end
    else
        wInit = cellfun(@(wIm) wIm(:,d),wInits,'UniformOutput',0);
        if coolDown
            wInit = PTDfromInit(X,sqrt(p'),wInit,param);
        end
        [w,r(d)] = PTDfromInit(X,c,wInit,param);
    end
    for m=1:M
        W{m}(:,d) = w{m};
        V{m}(:,d) = X{m}*w{m};
    end
    if d<D
        % deflate
        X = arrayfun(@(m) X{m} - (w{m}*(w{m}'*X{m}'))',1:M,'UniformOutput',false)';
    end
end

