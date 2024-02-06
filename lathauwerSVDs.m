function [U S] = lathauwerSVDs(A,k)

% lathauwerSVDs Tensor SVD from De Lathauwer et al. (2000)
%  [U S] = lathauwerSVDs(A,k)
%
%  EXAMPLE
%  A = rand(10,10,10);
%  [U S] = lathauwerSVDs(A,3);

%  De Lathauwer, Lieven, Bart De Moor, and Joos Vandewalle. "A multilinear 
%   singular value decomposition." SIAM journal on Matrix Analysis and 
%   Applications 21.4 (2000): 1253-1278.

dims = size(A);
U = cell(numel(dims),1);
S = A;
for m=1:numel(dims)
    Au = munfold(A,m);
    [U{m},~,~] = svds(Au,k);
    S = mmodeprod(S,m,U{m}');
end

    function Au = munfold(A,m)
        % m-mode unfolding of A
        rest = 1:ndims(A);
        rest(m) = [];
        Au = reshape(permute(A,[m rest]),size(A,m),[]);
    end

    function A = mfold(Au,m,dims)
        % "fold" m-mode unfolding Au back to A
        rest = dims;
        rest(m) = [];
        A = reshape(Au,[size(Au,1) rest]);
        order = 1:numel(dims);
        order = [order(2:m) 1 order(m+1:end)];
        A = permute(A,order);
    end

    function AmS = mmodeprod(A,varargin)
        % AmS = mmodeprod(A,m,S) m-mode product of tensor A with matrix S
        % AmS = mmodeprod(A,S) m-mode products with matrices in cell array S
        if nargin==3
            mm = varargin{1};
            Sm = varargin{2};
            Aum = munfold(A,mm);
            AmS = Aum'*Sm';
            dimsm = size(A);
            dimsm(mm) = size(Sm,1);
            AmS = mfold(AmS',mm,dimsm);
        elseif nargin==2
            Sm = varargin{1};
            AmS = A;
            for mm=1:numel(Sm)
                AmS = mmodeprod(AmS,mm,Sm{mm});
            end
        end
    end

end