clc;
clear;
close all;
%% define basis functions
syms X1 X2 U ;
assume([X1 X2 U] , 'real');

% phi(z)  z = [X ; u] for q learning
phi = [X1^2 X1*X2 X2^2 ...
       X1^4  X1^3*X2 X1^2*X2^2 X1*X2^3 X2^4 ...
       X1^6  X1^5*X2 X1^4*X2^2 X1^3*X2^3 X1^2*X2^4 X1*X2^5 X2^6 ...
       X1*U X2*U X1^3*U X2^3*U X1^5*U X2^5*U ...
       X1^4*X2*U X1*X2^4*U X1^2*X2*U X1*X2^2*U U^2]';
phi([2     4     5     7     9    10    11    12    14    15    17    18    19  20    21    22    24]) = [];  

% Q = W'*phi for q learning
Ws = sym('Ws' , [size(phi , 1) , 1]);
assume(Ws , 'real') ;

Q = Ws'*phi ;

dotQ = diff(Q , U) ;
Us = solve(dotQ , U) ;

uFCN  = matlabFunction(Us);
phiFCN = matlabFunction(phi); 

nP = 500 ;
M  = 100 ;
W = zeros(size(phi , 1) , nP); W(end , 1) = 1 ;
R = 1 ; Q = [0 0; 0 1] ;

for j = 1:nP
    
    PHI = [] ; 
    SAI = [] ;
    for t = 1:M
        xt = [-1; 1];
        ut = uFCN(W(6 , j),W(7 , j),W(8 , j),W(9 , j), xt(1) , xt(2));
        
        ut = ut + 0.01*randn ;
        
        xt1 = [xt(2); -1*xt(1)*((pi/2) + atan(5*xt(1))) - 5*xt(1)^2/(2*(1+25*xt(1)^2)) + 4*xt(2) + 3*ut] ;
        
        ut1 = uFCN(W(6 , j),W(7 , j),W(8 , j),W(9 , j), xt1(1) , xt1(2));
        
        PHI = [PHI ; phiFCN(ut , xt(1) , xt(2))'-phiFCN(ut1 , xt1(1) , xt1(2))'] ;
        SAI = [SAI ; xt'*Q*xt + ut'*R*ut] ;  
    end
    W(: , j+1) = PHI\SAI ;
    
    disp(['Iteration(' num2str(j) ')']);
    
    if norm(W(: , j+1)-W(: , j)) < 1e-4
        break;
    end
end

Woptimal = W(: , j+1)

%% simulation

Tf = 10 ;
Ts = 1 ;
t = 0:Ts:Tf ;
Nt = numel(t) ;
n = 2 ;

x = zeros(n , Nt) ;  x(: , 1) = randn(n,1) ;
u = zeros(1 , Nt) ;

Cost = x(: , 1)'*Q*x(: , 1) ;
for t = 1:Nt-1
    
    u(t) = uFCN(Woptimal(6) ,Woptimal(7) , Woptimal(8) ,Woptimal(9) , x(1 , t) , x(2 , t));
    
    x(: , t+1)= [x(2, t); -1*x(1, t)*((pi/2) + atan(5*x(1, t))) - 5*x(1, t)^2/(2*(1+25*x(1, t)^2)) + 4*x(2, t) + 3*u(t)] ;
    
    Cost = Cost + x(: , t+1)'*Q*x(: , t+1)+u(t)'*R*u(t) ;
end

Cost
%% plot results

Fig = figure(1) ;
Fig.Color = [1 1 1];
subplot(211);
plot(t , x , 'linewidth' , 3) ; hold on
grid on
xlabel('time (sec)' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
ylabel('x' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
title('Nonlinear LQR' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
legend('x1','x2');
% xlim([0 5])


Fig = figure(1) ;
Fig.Color = [1 1 1];
subplot(212);
plot(t , u , 'linewidth' , 3) ; hold on
grid on
xlabel('time (sec)' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
ylabel('u' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
legend('u');
% xlim([0 5])


