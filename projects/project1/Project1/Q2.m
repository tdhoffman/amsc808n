function Q2()
close all
% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];

% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
A = A2016;
% remove all rows with missing data
A(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)),:) = [];
% select CA, OR, WA, NJ, NY counties
ind = find((A(:,1)>=6000 & A(:,1)<=6999)); % ...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
A = A(ind,:);

[n,dim] = size(A);

% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

% select max subset of data with equal numbers of dem and gop counties
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
[n,dim] = size(A);
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

% set up data matrix and visualize original data
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 5; % Bachelor Rate
i3 = 6; % Unemployment Rate
ixx = [i1, i2, i3];
% visualize([], X, idem, igop, str(ixx), 'Original Data');

% rescale data to [0,1]
XX = X(:,ixx); % data matrix
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];

% set up optimization problem
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w0 = [-1;-1;1;1];
fun = @(I,Y,w)fun0(I,Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v)Hvec0(I,Y,w,v,lam);

% Get initial guess with SINewton
[w0,f,gnorm] = SINewton(fun,gfun,Hvec,Y,w0);

fprintf('w = [%d,%d,%d], b = %d\n',w0(1),w0(2),w0(3),w0(4));
visualize(w0, XX, idem, igop, str(ixx), 'SINewton');

% Set up and solve stochastic gradient descent problem
lam = 0.01; %Tikhonov regularization
sgdGradi = @(x,i) (-Y(i,:).*exp(-Y(i,:)*x))./(1 + exp(-Y(i,:)*x)) + n*lam*x';
wSGD = SGD(w0, sgdGradi, 64);
visualize(wSGD, XX, idem, igop, str(ixx), 'Stochastic GD');

bszs = 64:2:1024;
N = 1000; % number of runs to average over
avgfs = zeros(bszs,1);
for i = 1:length(bszs)
    fs = zeros(N,1);
    for niter = 1:N
        t0 = tic;
        wSGD = SGD(w0, sgdGradi, 64);
        tf = toc(t0);
        fs(niter) = f;
    end
    avgfs(i) = mean(fs);
end

lams = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5];
for i = 1:length(lams)
    sgdGradi = @(x,i) (-Y(i,:).*exp(-Y(i,:)*x))./(1 + exp(-Y(i,:)*x)) + n*lam*x';
    wSGD = SGD(w0, sgdGradi, 64);
end
end

function fig = visualize(w, XX, idem, igop, labs, varargin)
if nargin > 2
    titlestr = varargin{1};
end
% get bbox values
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));

fig = figure;
hold on; grid on;
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(labs(1),'Fontsize',fsz);
ylabel(labs(2),'Fontsize',fsz);
zlabel(labs(3),'Fontsize',fsz);
title(titlestr);

if ~isempty(w)
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
end
end

function f = fun0(I,Y,w,lam)
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end

function g = gfun0(I,Y,w,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end

function Hv = Hvec0(I,Y,w,v,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end