%jkbeta  Plots of Bessel functions of first kind Jk(beta)
%        Jk(beta)=(1/pi)*int_0^{pi} cos(beta*sin(mu)-k*mu)*dmu

%        11-01-04, 11-01-01, P. Mathys

beta = (0:.02:20)';
k = [0:8];
Jk = besselj(k,beta);          %Jn(:,k+1) is k-th order Bessel fun
plot(beta,Jk),grid
%axis([0 20 -0.4 1.0])
axis([0 10 -0.4 1.0])
[Jkmx,ix] = max(Jk);           %Maximum values
text(1,0.9,'J_0(\beta)')
for i=2:max(k)+1
  text(beta(ix(i))+0.1,Jkmx(i)+0.04,['J_{' int2str(k(i)) '}(\beta)'])
end
title('Plots of Bessel Functions of the First Kind')
xlabel('\beta'),ylabel('J_k(\beta)')

figure(gcf)
