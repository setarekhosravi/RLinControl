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

%% Algorithm Parameters
nP = 50; % Maximum number of policy iterations
lambda = 0.9; % Lambda parameter for weighted update

K = zeros(nP, n); % Storing policies
K(1, :) = [0.4, 0.5, 0.6]; % Initial policy (must be feasible)
P = cell(nP, 1); % Storing value functions
P{1} = zeros(n); % Initial value function

tol = 1e-6; % Convergence tolerance
converged = false;

% Convergence metrics
delta_K_values = zeros(nP, 1);
delta_P_values = zeros(nP, 1);

for j = 1:nP
    % Step 1: Policy Improvement (Update K)
    K_j = K(j, :);
    P_j = P{j};
    K(j+1, :) = (R + B' * P_j * B)^(-1) * (B' * P_j * A);
    
    % Step 2: Value Function Update (Solve for P_{j+1})
    options = optimoptions('fsolve', 'Display', 'off', 'TolFun', 1e-8);
    cost = @(P_vec) lambda_PI(P_vec, A, B, K(j+1, :), Q, R, P_j, lambda);
    P_next_vec = fsolve(cost, P_j(:), options);
    P{j+1} = reshape(P_next_vec, size(A));
    
    % Compute Convergence Metrics
    delta_K = norm(K(j+1, :) - K(j, :));
    delta_P = norm(P{j+1} - P{j}, 'fro'); % Frobenius norm for matrices
    delta_K_values(j) = delta_K;
    delta_P_values(j) = delta_P;
    
    fprintf('Iteration %d: ||K_{j+1} - K_j|| = %.8f, ||P_{j+1} - P_j|| = %.8f\n', ...
        j, delta_K, delta_P);
    
    % Convergence Check
    if delta_K < tol && delta_P < tol
        converged = true;
        break;
    end
end

% Final Output
if converged
    fprintf('Algorithm converged after %d iterations.\n', j);
else
    fprintf('Algorithm did not converge within %d iterations.\n', nP);
end

disp(['K Algorithm = ', num2str(K(j+1, :))]);
disp(['K LQR = ', num2str(K_lqr)]);

%% Plot Results

% Plot Policy Gains
figure(1);
for i = 1:3
    subplot(2, 2, i);
    plot(1:j+1, K(1:j+1, i), 'linewidth', 3); hold on;
    grid on;
    xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
    ylabel(['K', num2str(i)], 'fontSize', 14, 'fontWeight', 'Bold');
    title(['K', num2str(i)], 'fontSize', 14, 'fontWeight', 'Bold');
end

% Plot Convergence Metrics
figure(2);
subplot(2, 1, 1);
plot(1:j, delta_K_values(1:j), 'r', 'linewidth', 2); hold on;
grid on;
xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
ylabel('||K_{j+1} - K_j||', 'fontSize', 14, 'fontWeight', 'Bold');
title('Policy Convergence', 'fontSize', 16, 'fontWeight', 'Bold');

subplot(2, 1, 2);
plot(1:j, delta_P_values(1:j), 'b', 'linewidth', 2); hold on;
grid on;
xlabel('Iteration', 'fontSize', 14, 'fontWeight', 'Bold');
ylabel('||P_{j+1} - P_j||', 'fontSize', 14, 'fontWeight', 'Bold');
title('Value Function Convergence', 'fontSize', 16, 'fontWeight', 'Bold');

%% Lambda Policy Iteration Cost Function
function z = lambda_PI(P_vec, A, B, K, Q, R, P_j, lambda)
    P = reshape(P_vec, size(A)); % Reshape vector back into matrix form
    K_mat = reshape(K, size(B, 2), size(A, 1));
    
    % Compute the lambda-weighted equation
    M = (1 - lambda) * (A - B * K_mat)' * P_j * (A - B * K_mat) + ...
        lambda * (A - B * K_mat)' * P * (A - B * K_mat) + ...
        K_mat' * R * K_mat + Q - P;
    
    z = M(:); % Flatten the matrix for optimization
end
