function [x, gnorm] = stoLBFGS(x0, func, gfun, ng, nh, M)
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
tol = 1e-10;
m = 5; % the number of steps to keep in memory

n = size(x0,1);
s = zeros(4,m);
y = zeros(4,m);
rho = zeros(1,m);
gnorm = zeros(1,1000);

x = x0;
idxg = randi(n, ng, 1); %generate random batch
g = gfun(x, -1);
gnorm(1) = norm(g);

% first do steepest decent step
a = linesearch(x,-g,g,@(x) func(x,-1),eta,gam,jmax);
xnew = x - a*g;
idxg = randi(n, ng, 1); %generate random batch
gnew = gfun(xnew, -1);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
gnorm(2) = nor;
iter = 1;
while nor > tol
    idxg = randi(n, ng, 1); %generate random batch
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    a = a*gam;
%     a = 1/iter; % a = a*gam;
%     Backtracking line search
%     [a,j] = linesearch(x,p,g,@(x) func(x, -1),eta,gam,jmax);
%     if j == jmax
%         p = -g;
%         [a, ~] = linesearch(x,p,g,@(x) func(x, -1),eta,gam,jmax);
%     end
    
    step = a*p;
    xnew = x + step;
    gnew = gfun(xnew, -1);
    
    if mod(iter, M) == 0        
        idxh = randi(n, nh, 1);
        % Remove eldest pairs
        s = circshift(s,[0,1]); 
        y = circshift(y,[0,1]);
        rho = circshift(rho,[0,1]);
        
        % Update s and y
        s(:,1) = step;
        y(:,1) = gfun(xnew, idxh) - gfun(x, idxh);
        rho(1) = 1/(step'*y(:,1));
    end
    
    x = xnew;
    g = gnew;
    
    nor = norm(g);
    iter = iter + 1;
    gnorm(iter+1) = nor;
    fprintf('nor = %f, a = %f\n', nor, a);
end
fprintf('SL-BFGS: %d iterations, norm(g) = %d\n',iter,nor);
gnorm(iter+1:end) = [];
end

function [a,j] = linesearch(x,p,g,func,eta,gam,jmax)
    a = 1;
    f0 = func(x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        f1 = func(xtry);
        if f1 <= f0 + a*aux %must add an equals sign here otherwise never terminates
            break;
        else
            a = a*gam;
        end
    end
end