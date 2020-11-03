function x = SGD(x0, gfun, bsz)
% Input: x0   = initial guess
%        gfun = gradient of function accepting indices to report at
%        bsz  = batch size
xp = x0;
n = size(xp,1);
maxiter = 200;
for iter = 1:maxiter
    idx = randi(n, bsz, 1); %generate random batch
    alpha = 1/iter; %decrease step size
    gp = sum(gfun(xp, idx))/bsz;
    x = xp - alpha*gp;
end
end