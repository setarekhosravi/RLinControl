clc;
clear;
close all;

%% System Definition
A = [0.9065 0.0816 -0.0005; 0.0743 0.9012 -0.0007; 0 0 0.1327];
B = [-0.0027 -0.0068 1]';

n = size(A, 1);

R = 1;
Q = eye(n);
E = eye(n);
S = zeros(n, 1);

[P_lqr, K_lqr, ~] = idare(A, B, Q, R, S, E);

%% Value Iteration Algorithm
nP = 100; % Number of policy iterations
nGPI = 10; % Number of gradient policy iterations for P update
K = zeros(nP, n);
K(1, :) = [0.4, 0.5, 0.6]; % Initial policy (should be feasible)
P = cell(nP, 1);
P{1} = zeros(n);

% Convergence metrics
delta_K_values = zeros(nP, 1);
delta_P_values = zeros(nP, 1);

tic
for j = 1:nP
    % Step 1: Update Value Function (P_{j+1})
    PP = P{j};
    for i = 1:nGPI
        PP = (A - B * K(j, :))' * PP * (A - B * K(j, :)) + Q + K(j, :)' * R * K(j, :);
    end
    P{j+1} = PP;
    
    % Step 2: Update Policy (K_{j+1})
    K(j+1, :) = (R + B' * P{j+1} * B)^(-1) * (B' * P{j+1} * A);
    
    % Compute Convergence Metrics
    delta_K = norm(K(j+1, :) - K(j, :));
    delta_P = norm(P{j+1} - P{j}, 'fro'); % Frobenius norm for matrices
    delta_K_values(j) = delta_K;
    delta_P_values(j) = delta_P;
    
    disp(['Iteration(', num2str(j), ')']);
    
    % Convergence Check
    if delta_K < 1e-6
        break;
    end
end
disp(['Elapsed Time = ', num2str(toc)]);

disp(['K LQR = ', num2str(K_lqr)]);
disp(['K VI = ', num2str(K(j+1, :))]);

%% Plot Policy Gains
Fig = figure(1);
Fig.Color = [1, 1, 1];

for i = 1:3
    subplot(2, 2, i);
    plot(1:j+1, K(1:j+1, i), 'linewidth', 3);
    hold on;
    grid on;
    xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
    ylabel(['K', num2str(i)], 'fontSize', 14, 'fontWeight', 'Bold');
    title(['K', num2str(i)], 'fontSize', 14, 'fontWeight', 'Bold');
end

%% Plot Convergence Metrics
Fig2 = figure(2);
Fig2.Color = [1, 1, 1];

subplot(2, 1, 1);
plot(1:j, delta_K_values(1:j), 'r', 'linewidth', 2);
grid on;
xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
ylabel('||K_{j+1} - K_j||', 'fontSize', 14, 'fontWeight', 'Bold');
title('Policy Convergence', 'fontSize', 16, 'fontWeight', 'Bold');

subplot(2, 1, 2);
plot(1:j, delta_P_values(1:j), 'b', 'linewidth', 2);
grid on;
xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
ylabel('||P_{j+1} - P_j||', 'fontSize', 14, 'fontWeight', 'Bold');
title('Value Function Convergence', 'fontSize', 16, 'fontWeight', 'Bold');
