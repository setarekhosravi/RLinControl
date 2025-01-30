clc;
clear;
close all;

%% System Definition
n = 3; % State dimension
m = 1; % Control dimension

% Generate synthetic data (replace with real sampled data if available)
num_samples = 1000;
A_true = [0.9065 0.0816 -0.0005; 0.0743 0.9012 -0.0007; 0 0 0.1327]; % True A
B_true = [-0.0027; -0.0068; 1]; % True B
x_samples = randn(n, num_samples); % Random states
u_samples = randn(m, num_samples); % Random controls
x_next_samples = A_true * x_samples + B_true * u_samples + 0.01 * randn(n, num_samples); % Next states with noise

%% Step 1: Estimate Dynamics (A, B) from Sampled Data
% Solve x_{k+1} = A*x_k + B*u_k using least squares
X = x_samples(:, 1:end-1); % Current states x_k
U = u_samples(:, 1:end-1); % Control inputs u_k
X_next = x_next_samples(:, 1:end-1); % Next states x_{k+1}

% Construct regression problem
Theta = [X; U]; % Combine state and control inputs
W = X_next; % Target next state

% Solve for [A, B] using least squares
AB = W / Theta; % [A, B] = W * pinv(Theta)
A_est = AB(:, 1:n); % Extract A
B_est = AB(:, n+1:end); % Extract B

disp('Estimated A:');
disp(A_est);
disp('Estimated B:');
disp(B_est);

%% Step 2: Policy Iteration with Estimated Dynamics
R = 1; % Control cost
Q = eye(n); % State cost

nP = 50; % Number of iterations
K = zeros(nP, n); % Policy (control gains)
K(1, :) = [0.4, 0.5, 0.6]; % Initial policy guess

P = cell(nP, 1); % Cost-to-go matrices
P{1} = zeros(n);

for j = 1:nP
    % Step 1: Solve for P_{j+1} using Riccati equation
    P{j+1} = (Q + K(j, :)' * R * K(j, :) + (A_est - B_est * K(j, :))' * P{j} * (A_est - B_est * K(j, :)));
    
    % Step 2: Update Policy
    K(j+1, :) = (R + B_est' * P{j+1} * B_est)^(-1) * (B_est' * P{j+1} * A_est);

    % Display iteration progress
    disp(['Iteration ', num2str(j), ' Complete']);
end

disp('Final Policy:');
disp(K(end, :));
