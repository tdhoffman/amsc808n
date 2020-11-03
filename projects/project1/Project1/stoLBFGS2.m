function x = stoLBFGS2(x0, func, gfun, ng, nh, M)
% Input: x0   = initial guess
%        func = function in question
%        gfun = gradient of function accepting indices to report
%        m    = number of previous steps to keep in memory
%        ng   = batch size for stochastic gradient calculation
%        nh   = batch size for stochastic Hessian calculation
%        M    = number of iterations between Hessian updates

tol = 1e-10; %stopping criterion

% Line search parameters
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor

% initialize arrays
n = size(x0,1);
s = zeros(n,m);
y = zeros(n,m);
rho = zeros(1,m);
x = x0;
idxg = randi(n, ng, 1); %generate random batch
g = gfun(x, -1);
n = size(x, 1);

% first do steepest decent step
a = linesearch(x,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
gnew = gfun(xnew, -1);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
iter = 1;
while nor > tol
    idxg = randi(n, ng, 1); %generate random batch
    idxh = randi(n, nh, 1); %generate random batch
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    [a, j] = linesearch(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a, ~] = linesearch(x,p,g,func,eta,gam,jmax);
    end
    
    step = a*p;
    xnew = x + step;
    gnew = gfun(xnew, -1);
    
    if mod(iter, M) == 0        
        % Remove eldest pairs
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        
        % Update s and y
        s(:,1) = step;
        y(:,1) = gfun(xnew, -1) - gfun(x, -1);
        rho(1) = 1/(step'*y(:,1));
    end
    
    x = xnew;
    g = gnew;
    nor = norm(g);
    iter = iter + 1;
    fprintf('iteration %d complete, norm = %f, a = %.9f\n', iter, nor, a);
end
end

function [a,j] = linesearch(x,p,g,func,eta,gam,jmax)
    a = 1;
    f0 = func(x, -1);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = func(xtry, -1);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end