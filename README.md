# PTD-CCA: Sparse Tensor Canonical Correlation Analysis (STCCA) via penalised tensor decomposition

PTD-CCA is an unsupervised dimensionality reduction method for 2 or more views/data modalities. The algorithm is a straightforward extension of the penalised matrix decomposition CCA (PMD-CCA) proposed by Witten et al. (2009) to more than 2 views and maximises the "higher-order covariance" between the linear projections `X_m*w_m` where each `X_m` is data matrix and `w_m` a vector of coefficients. It reduces to PMD-CCA if there are just 2 views.

## EXAMPLE

Set up some synthetic data
```MATLAB
a = [ones(20,1); -ones(20,1); zeros(60,1)];
b = [zeros(60,1); -ones(20,1); ones(20,1)];
c = [ones(20,1); zeros(60,1); -ones(20,1)];
d = [-ones(10,1); ones(10,1); zeros(60,1); -ones(10,1); ones(10,1)];
Z = rand(100,4); Z = Z./sum(Z,2);
X1 = normrnd(Z(:,1)*a',0.1);
X2 = normrnd(Z(:,2)*b',0.1);
X3 = normrnd(Z(:,3)*c',0.1);
X4 = normrnd(Z(:,4)*d',0.1);
X = {X1;X2;X3;X4};
```
Optimise the sparsity parameter `c`
```MATLAB
optC = tunePTDCCA(X,'initType',"matrix");
```
Here we are avoiding calculating the covariance tensor of `X` by setting the option `'initType'` to `"matrix"` to save time.

Run `PTDCCA` and compare the model to the ground truth.
```MATLAB
W = PTDCCA(X,optC);
wtrue = [a,b,c,d];
for m=1:4
    subplot(2,4,m);bar(wtrue(:,m));title(sprintf('True w_%d',m))
    subplot(2,4,4+m);bar(W{m});title(sprintf('Inferred w_%d',m))
end
```

## References
Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.
