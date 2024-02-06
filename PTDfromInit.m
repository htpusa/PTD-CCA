function [w,obj] = PTDfromInit(X,c,wInit,param)

maxIter = param.maxIter;
eps = param.eps;

M = numel(X);

w = cellfun(@(wm) projectL2(wm,1),wInit,'UniformOutput',false);
Xw = cell2mat(cellfun(@mtimes,X,w,'UniformOutput',false)');

%obj = sum(prod(Xw,2));
obj = 0;
iter = 0;
improvement = 42;

while improvement>eps && iter<maxIter

    for m=1:M
        Xwom = Xw; Xwom(:,m) = [];
        w{m} = projectL1L2(X{m}'*prod(Xwom,2),c(m));
        Xw(:,m) = X{m}*w{m};
    end
    objNew = sum(prod(Xw,2));
    improvement = (objNew-obj)/abs(obj);
    obj = objNew;
    iter = iter + 1;
end

if iter==maxIter
    %warning('PTDfromInit reached maximum number of iterations')
end

    function xProj = projectL2(x,c)
        xProj = c*x/norm(x,2);
    end

    function xProj = projectL1L2(x,c)
        % PROJECTL1L2 L_1-L_2-projection from Witten et al. 2009
        %  xProj = projectL1L2(x,c)
        %
        %  EXAMPLE
        %  x = rand(10,1);
        %  xProj = projectL1L2(x,2);
        
        %  Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. 
        %    "A penalized matrix decomposition, with applications to sparse 
        %    principal components and canonical correlation analysis." 
        %    Biostatistics 10.3 (2009): 515-534.
        
        convergenceCrit = 1e-5;
        maxIterSub = 100;
        nMax = sum(max(abs(x))==abs(x));
        
        if sqrt(nMax)>=c % c<=1 or pathological duplicate max values
            [~,i] = max(abs(x));
            xProj = zeros(numel(x),1);
            xProj(i) = c*sign(x(i));
        else
            xProj = projectL2(x,1);
        end
        
        cont = norm(xProj,1) > c;
        deltaRange = [0 max(abs(x))];
        iterSub = 0;
        
        while cont
            delta = mean(deltaRange);
            xProj = projectL2(softThresh(x,delta),1);
            diff = norm(xProj,1) - c;
            if diff>0
                deltaRange(1) = delta;
            else
                deltaRange(2) = delta;
            end
            cont = abs(diff) > convergenceCrit && iterSub<=maxIterSub;
            iterSub = iterSub+1;
        end
            function y = softThresh(x,c)
                y = abs(x)-c;
                y(y<0) = 0;
                y = sign(x) .* y;
            end
    end

end