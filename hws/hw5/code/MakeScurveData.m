function MakeScurveData(amt, fname)
tt = [-1:0.1:0.5]*pi; uu = tt(end:-1:1); hh = [0:0.1:1]*5;
xx = [cos(tt) -cos(uu)]'*ones(size(hh));
yy = ones(size([tt uu]))'*hh;
zz = [sin(tt) 2-sin(uu)]'*ones(size(hh));
cc = [tt uu]' * ones(size(hh));
data3 = [xx(:),yy(:),zz(:)];
data3 = data3 + amt*randn(size(data3,1), 3);
figure;
hold on;
plot3(data3(:,1),data3(:,2),data3(:,3),'.','Markersize',20);
daspect([1,1,1]);
set(gca,'fontsize',16);
view(3);
grid
save(fname,'data3');
end
 
  
