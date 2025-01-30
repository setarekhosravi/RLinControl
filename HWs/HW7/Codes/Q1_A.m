clc;
clear;
close all;
%%
A = [0.9065 0.0816 -0.0005;0.0743 0.9012 -0.0007;0 0 0.1327];
B = [-0.0027 -0.0068 1]';

n = size(A , 1) ;

R = 1 ;
Q = eye(n);
E = eye(n);
S = zeros(n , 1) ;

[P_lqr , K_lqr , L] = idare(A , B , Q , R , S , E)
%%
Ts = 0.1;
Tf = 20; 
t = 0:Ts:Tf ;
Nt = numel(t) ;

x = zeros(n , Nt) ;  x(: , 1) = [10 -10 -3]'; %randn(4,1) ;
u = zeros(1 , Nt) ;

Cost = x(: , 1)'*Q*x(: , 1) ;
for k = 1:Nt-1
    u(k) = -K_lqr*x(: , k) ;
    
    x(: , k+1) = A*x(: , k)+B*u(k) ;
    
    Cost = Cost + x(: , k+1)'*Q*x(: , k+1)+u(k)'*R*u(k) ;
end

%% plot results

Fig = figure(1) ;
Fig.Color = [1 1 1];
subplot(211);
plot(t , x , 'linewidth' , 3) ; hold on
grid on
xlabel('time (sec)' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
ylabel('x' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
title('simulation of the closed loop control system' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
legend('x1','x2','x3');
xlim([0 5])


Fig = figure(1) ;
Fig.Color = [1 1 1];
subplot(212);
plot(t , u , 'linewidth' , 3) ; hold on
grid on
xlabel('time (sec)' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
ylabel('u' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
legend('u');
xlim([0 5])