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

Run `PTDCCA` with "intermediate" sparsity and compare the model to the ground truth.
```MATLAB
W = PTDCCA(X,0.5);
wtrue = [a,b,c,d];
figure
for m=1:4
    subplot(2,4,m);bar(wtrue(:,m));title(sprintf('True w_%d',m))
    xlabel('variable');ylabel('coefficient')
    subplot(2,4,4+m);bar(W{m});title(sprintf('Inferred w_%d',m))
end
```

Sparsity can also be set for each view separately:
```MATLAB
c = [0.05,0.25,0.75,1];
W = PTDCCA(X,c);
for m=1:4
    subplot(1,4,m);bar(W{m});title(sprintf('c = %.2f',c(m)))
    xlabel('variable');ylabel('coefficient')
end
```

To calculate multiple canonical variable tuples, use the name-value input `D`
```MATLAB
W = PTDCCA(X,0.5,'D',3);
```

If you've ran the examples above, you may have noticed the function takes some time to return. Most of the running time is in fact spent calculating the cross-covariance tensor which is used to initialise the algorithm. This can be avoided by using a random initialisation instead:
```MATLAB
W = PTDCCA(X,0.5,'initType','random');
```
The covariance tensor also takes up a lot of memory, and if the dimensions of the data are high enough, might exceed the largest allowed array size. If this happens, `PTDCCA` defaults to the random initialisation.

## References
Witten, Daniela M., Robert Tibshirani, and Trevor Hastie. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." Biostatistics 10.3 (2009): 515-534.
