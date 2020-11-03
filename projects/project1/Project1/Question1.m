function Question1()
close all
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];
%% select CA, OR, WA, NJ, NY counties
ind = find((A(:,1)>=6000 & A(:,1)<=6999)  ...  %CA
  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
  | (A(:,1)>=34000 & A(:,1)<=34999)); % ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
A = A(ind,:);

[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties
ngop = length(igop);
ndem = length(idem);
if ngop > ndem
    rgop = randperm(ngop,ndem);
    Adem = A(idem,:);
    Agop = A(igop(rgop),:);
    A = [Adem;Agop];
else
    rdem = randperm(ndem,ngop);
    Agop = A(igop,:);
    Adem = A(idem(rdem),:);
    A = [Adem;Agop];
end  
[n,dim] = size(A)
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% set up data matrix and visualize
close all
figure;
hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 5; % Bachelor Rate
i3 = 6; % Unemployment Rate
plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',20);
plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% rescale data to [0,1] and visualize
figure;
hold on; grid;
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);

%% set up optimization problem
C = 0;

[n,d] = size(XX);

% First run SINewton to get initial x_0 guess
lam = 0.01;
Y = (label*ones(1,d + 1)).*[XX,ones(n,1)];
w = [-1;-1;1;1];
fun = @(I,Y,w)fun0(I,Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v)Hvec0(I,Y,w,v,lam);
[x_0,~,~] = SINewton(fun,gfun,Hvec,Y,w,-1);

% Changing dimension to allow slack variables in x
x_0(n+d+1) = 0;

% A matrix input to ASM
A = cell2mat({Y, eye(n); zeros(n,d+1), eye(n)});

% b vector input to ASM
b = ones(2 * n, 1);
for i = n+1:2*n
    b(i) = 0;
end

% functions are defined at bottom
gfun = @(x) g_soft_margin(x, d, n, C);
hfun = @(x) h_soft_margin(x, d, n, C);

% initial active set is empty
W = [];

bulksz = 100;
gnorms = zeros(bulksz,5000);
times = zeros(bulksz,1);
for j = 1:bulksz
    % run ASM and retrieve f and gnorm at each iterate
    t0 = tic;
    [xiter, ~, gnorm] = ASM(x_0, gfun, hfun, A, b, W);
    times(j) = toc(t0);
    [~,iters] = size(xiter);
    gnorms(j,:) = gnorm;
end
avged = mean(gnorms);

w_final = xiter(:,iters);
w = w_final(1:4);

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

%%
figure;
hold on;
grid;
niter = length(avged);
plot(0:niter-1,avged,'Linewidth',2);
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| grad f ||','Fontsize',fsz);

figure;
hold on;
grid;
plot(0:bulksz-1,times,'Linewidth',2);
set(gca,'Fontsize',fsz);
xlabel('trial number','Fontsize',fsz);
ylabel('runtimes','Fontsize',fsz);
end

%%
function f = f_soft_margin(x, d, n, C)
q = eye(d+n+1);
for i = n+1:d+n+1
    q(i,i) = 0;
end

v = ones(1,d+n+1);
for i = 1 : n+1
    v(i) = 0;
end
v = C * v;

f = 0.5 * x' * q * x + v * x;


end

function g = g_soft_margin(x, d, n, C)
q = eye(d+n+1);
for i = n+1:d+n+1
    q(i,i) = 0;
end

v = ones(1,d+n+1);
for i = 1 : n+1
    v(i) = 0;
end
v = C * v;

g = q * x + v';
end

function h = h_soft_margin(x, d, n, C)
q = eye(d+n+1);
for i = n+1:d+n+1
    q(i,i) = 0;
end

h = q;
end


%%
function f = fun0(I,Y,w,lam)
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end
%%
function g = gfun0(I,Y,w,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end
%%
function Hv = Hvec0(I,Y,w,v,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end









