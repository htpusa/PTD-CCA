function [W r V c] = PTDCCApath(X,varargin)

% PTDCCApath
%   [W r V] = PTDCCApath(X)
%   Variant of PTDCCA that produces a regularisation path ie a sequence of
%   models for a sequence of regularisation parameters. This can be used
%   to efficiently compute models for parameter tuning.
%
%   INPUTS:
%   X           -   Mx1 cell where M is the number of views and X{M} is a
%                       NxPm matrix with N rows corresponding to samples and
%                       Pm columns to variables
%   OPTIONAL INPUTS:
%   'D'         -   double, the number of canonical variable tuples
%                       (default: 1)
%   'L'         -   double, length of the sequence ie how many different
%                       c values are used (default: 20)
%   'c'         -   vector, the sequence of c values to use
%                       should be in ascending order
%                       (default: linspace(0,1,L))
%   'initType'  -  "tensor" or "random", how to initialise the algorithm
%                       (default: "tensor")
%                       "tensor" uses the singular vectors of the
%                       cross-covariance tensor. Probably the best option but
%                       costly for larger data
%                       "random" tries several random starts
%   'CCten'     -   P1xP2x...xPm tensor, the cross-variance tensor of the data
%                       in X
%                       If the algorithm is run multiple times, time can be
%                       saved by calculating the tensor offline
%   'maxIter'   -   maximum number of iterations (default: 1000)
%   'eps'       -   stopping criterion threshold (default: 1e-10)
%   'rStarts'   -   how many random starts are tried (default: 5)
%
%   OUTPUTS:
%   'W'         -   Mx1 cell where W{m} is an Pmx100xD matrix
%                       W{m}(:,l,d) is the dth coefficient vector
%                       calculated with c(l)
%   'r'         -   100xD vector of objective values
%   'V'         -   Mx1 cell where V{m} is a Nx100xD matrix
%                       V{m}(:,l,d) is the dth variable vector
%                       calculated with c(l)
%   'c'         -   Lx1 vector of regularisation parameter values
%
%   EXAMPLE:
%      load carbig;
%      data = [Displacement Horsepower Weight Acceleration MPG Cylinders Model_Year];
%      nans = sum(isnan(data),2) > 0;
%      X = {data(~nans,1:2); data(~nans,3:4); data(~nans,5:end)};
%      [W r V] = PTDCCApath(X);

%   Author: T.Pusa, 2024

% default parameters
L = 20;
D = 1;
c = [];
param.maxIter = 1000;
param.eps = 1e-10;
CCten = [];
init = "tensor";
rInits = 5;

if ~isempty(varargin)
    if rem(size(varargin, 2), 2) ~= 0
		error('Check optional inputs.');
    else
        for i = 1:2:size(varargin, 2)
            switch varargin{1, i}
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

if isempty(c)
    c = linspace(0,1,L)';
    c01 = c;
    c = c.*(sqrt(p')-1)+1;
else
    c = c(:);
    c01 = c;
    c = c.*(sqrt(p')-1)+1;
    L = numel(c01);
end

W = arrayfun(@(pm) zeros(pm,L,D),p,'UniformOutput',false);
r = zeros(L,D);
V = arrayfun(@(n) zeros(n,L,D),N*ones(M,1),'UniformOutput',false);

X = cellfun(@(Xm) Xm - mean(Xm,1),X,'UniformOutput',false);

for d=1:D
    if init=="random"
        for rI=1:rInits
            wInit = arrayfun(@(m) rand(p(m),D)-0.5,1:M,'UniformOutput',0)';
            [wtmp,rtmp] = PTDfromInit(deflate(X,W,L,d),c(L,:),wInit,param);
            if rtmp>r(L,d)
                w = wtmp;
                r(L,d) = rtmp;
            end
        end
    else
        wInit = cellfun(@(wIm) wIm(:,d),wInits,'UniformOutput',0);
        [w,r(L,d)] = PTDfromInit(deflate(X,W,L,d),c(L,:),wInit,param);
    end
    for m=1:M
        W{m}(:,L,d) = w{m};
        V{m}(:,L,d) = X{m}*w{m};
    end
    for l=L-1:-1:1
        [w,r(l,d)] = PTDfromInit(deflate(X,W,l,d),c(l,:),w,param);
        for m=1:M
            W{m}(:,l,d) = w{m};
            V{m}(:,l,d) = X{m}*w{m};
        end
    end
end

c = c01;

    function Xd = deflate(X,W,l,d)
        Xd = X;
        for dd=1:d-1
            Xd = arrayfun(@(m) Xd{m} - (W{m}(:,l,dd)*(W{m}(:,l,dd)'*Xd{m}'))',...
                1:M,'UniformOutput',false)';
        end
    end

end
