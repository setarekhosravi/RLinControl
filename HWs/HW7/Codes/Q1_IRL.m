clc;
clear;
close all;

%% System Definition
A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B = [0; 0; 1];

n = size(A, 1);

R = 1;
Q = eye(n);
%% Policy Iteration Algorithm
nP = 100; % Number of policy iterations
K = zeros(nP, n); % Storing policies
K(1, :) = [0.4, 0.5, 0.6]; % Initial policy (should be feasible)
P = cell(nP, 1);
P{1} = zeros(n);

% Convergence metrics
delta_K_values = zeros(nP, 1);
delta_P_values = zeros(nP, 1);

options = optimoptions('fmincon', 'Display', 'off');

for j = 1:nP
    % Step 1: Solve for P_{j+1}
    cost = @(P) PI(P, A, B, K(j, :), Q, R);
    [Ps, ~] = fmincon(cost, P{j}(:), [], [], [], [], [], [], [], options);
    P{j+1} = reshape(Ps, size(A));
    
    % Step 2: Update Policy (K_{j+1})
    K(j+1, :) = (inv(R) * B' * P{j+1});
    
    % Compute Convergence Metrics
    delta_K = norm(K(j+1, :) - K(j, :));
    delta_P = norm(P{j+1} - P{j}, 'fro'); % Frobenius norm for matrices
    delta_K_values(j) = delta_K;
    delta_P_values(j) = delta_P;
    
    disp(['Iteration(', num2str(j), ')']);
    disp(['K = ', num2str(K(j+1, :))]);
    disp(['P = ']);
    P{j+1}
    

    % Convergence Check
    if delta_K < 1e-6
        break;
    end
end

disp(['Final K PI = ', num2str(K(j+1, :))]);

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

%% Optimization Function
function z = PI(P, A, B, K, Q, R)
    P = reshape(P, size(A));
    M = (A - B * K)' * P + P * (A - B * K) + Q + K' * R * K;
    z = sum(abs(M(:)));
end
