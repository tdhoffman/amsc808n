function stardata()
N = 100; % # of data points along each tenticle
rmin = 0.1; % inner radius
rmax = 0.9; % outer radius
ntts = 6; % the number of petals
da = 2*pi/ntts; % step in angle
a0 = 0:da:(ntts-1)*da; % initial angles for the petals
r = linspace(rmin,rmax,N); % radial space
phi = zeros(N,ntts); % set of angles for the data points
pfun = @(r)r; % function for twisting petals
for j = 1 : ntts
    phi(:,j) = a0(j) + pfun(r);
end
l = N*ntts; % the number of data points
X = zeros(l,2); % the data
y = zeros(l,1); % the labels 
for j = 1 : ntts
    for k = 1 : N
        i = (j-1)*N + k;
        X(i,:) = [r(k)*cos(phi(k,j)),r(k)*sin(phi(k,j))] + 0.1*r(k)*randn(1,2);        
    end
    y((j-1)*N+1:j*N) = (-1)^(mod(j,2));
end
% permute the data so that they are randomly distributed among the petals
p = randperm(l); 
X = X(p,:); % finalize the data
y = y(p,:); % finalize the labels

%% graphics
fsz = 16;
close all
figure(1);
hold on;
iminus = find(y == -1);
plot(X(iminus,1),X(iminus,2),'Linestyle','none','Marker','s','color','k');
iplus = setdiff((1:l)',iminus);
plot(X(iplus,1),X(iplus,2),'Linestyle','none','Marker','<','color','b');
set(gca,'Fontsize',fsz);
xlabel('x_1','Fontsize',fsz);
ylabel('x_2','Fontsize',fsz);
daspect([1,1,1]);
axis tight
%%
save('stardata.mat','X','y');
fid = fopen('stardata.txt','w');
for i = 1 : 600
    fprintf(fid,'%d\t%d\t%d\n',X(i,1),X(i,2),y(i));
end
fclose(fid);
end

