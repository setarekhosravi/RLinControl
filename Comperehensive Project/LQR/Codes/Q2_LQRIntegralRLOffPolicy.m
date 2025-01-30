%% RL in Control Comprehensive Project
% @author: STRH/99411425

%% 
clc;
clear;
close all;
%% Define Dynamics
A = [0.5 1.5; 2 -2];
B = [1 4]';
C = [1 0]; % doesn't need

K0 = [1.5 0.75];
n = size(A,1);

Q = eye(n);
R = 1;

% disp("A rank:")
% disp(rank(obsv(sqrt(Q),A)))
%% Calculate optimal K and P with LQR command
[K_lqr, P_lqr] = lqr(A,B,Q,R);
disp("K derived from LQR: ")
disp(K_lqr)

disp("P derived from LQR: ")
disp(P_lqr)
%% LQR using IRL
nP = 50;
M = 8;
Ts = 0.001;
T = 1;
time = 0:Ts:T; Nt = numel(time);
K = zeros(nP, n); K(1,:) = K0;
P_cell = cell(nP , 1) ; P_cell{1} = zeros(n);

for j = 1:nP
    
    PHI = []; 
    SAI = [] ;
    
    for i = 1:M
        x = zeros(n , Nt) ; x(: , 1) = randn(n , 1) ;
        u = zeros(1 , Nt) ;
        r = zeros(1 , Nt) ;
        
        for k = 1:Nt-1
            u(k) = -K(j , :)*x(: , k)+0.01*rand ;
            x(: , k+1) = x(: , k) + Ts*(A*x(: , k)+B*u(k));
            r(k) = (x(:, k)'*Q*x(:, k)) + u(k)'*R*u(k) + 2*u(k)'*R*(u(k) + K(j , :)*x(:, k));
            
        end
        SAI = [SAI ; trapz(time , r)];
        PHI = [PHI ; ComputeXbar(x(: , 1))-ComputeXbar(x(: , k+1))] ;
    end
    
    Pbar = PHI\SAI ;
    P = ConvertPbarToP(Pbar) ;
    P_cell{j+1} = P;
    
    K(j+1 , :) = inv(R)*B'*P ;
    
    disp(['Iteration(' num2str(j) ')']);
    
    if norm(P_cell{j+1}-P_cell{j}, 'fro') < 1e-4
        break;
    end
end

disp(['K LQR = ' num2str(K_lqr)]);
disp(['K IRL = ' num2str(K(j+1 , :))]);

Fig = figure(1) ;
Fig.Color = [1 1 1];

for i = 1:2
    subplot(2,2,i);
    plot(1:j+1 , K(1:j+1 , i) , 'linewidth' , 3) ; hold on
    grid on
    xlabel('Itr' , 'fontSize' , 14 , 'fontWeight' , 'Bold');
    ylabel(['K' num2str(i)] , 'fontSize' , 14 , 'fontWeight' , 'Bold');
    title(['K' num2str(i)] , 'fontSize' , 14 , 'fontWeight' , 'Bold');
end


%% functions

function Xbar = ComputeXbar(X)
    X = X(:)'; 
    Xbar = [] ; 
    
    for i = 1:numel(X)
        Xbar = [Xbar X(i)*X(i:end)];
    end
end

function P = ConvertPbarToP(Pbar)

    P = [Pbar(1)   Pbar(2)/2    
         Pbar(2)/2 Pbar(3)];
end

