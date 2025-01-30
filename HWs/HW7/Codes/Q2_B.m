clear
clc
close all

%% Define System and Config
A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B2 = [0 0 1]';
B1 = [1 0 0]';

Q = eye(3); R = 1; beta = 5;

% Set the number of iterations
nP = 20;

n = size(A, 1); % order
K1 = zeros(nP, n);
K2 = zeros(nP, n);

% initial values policy
K1(1, :) = [0.4 0.5 0.6];
K2(1, :) = [0 0 0];

% behavior policy
K1b = K1(1, :);
K2b = K2(1, :);

%% Learning Loop
t = 10;
dt = 0.01; % sampling time
N = t / dt; % number of samples for integral


value = 0;
xu = [0 0 0];
xd = [0 0 0];

P = cell(nP, 1);
P{1} = zeros(n);

% Policy Iteration
for j = 1:nP
    
    x = zeros(3, N);
    x(:, 1) = [10; -10; -3];
    phi_j = [];
    sai_j = [];
    for t = 1:N
        ub = -K1b * x(:, t) + 0.01 * randn;
        db = K2b * x(:, t);
        
        % Compute target policies
        u_j = -K1(j, :) * x(:, t);
        d_j = K2(j, :) * x(:, t);
        
        % Compute the integrand from the Bellman equation:
        % (-Q(x) - u_j^T R u_j + β^2 d_j^T d_j - 2u_{j+1}^T R(u-u_j) + 2β^2 d_{j+1}^T(d-d_j))
        value = value + dt * (-x(:, t)' * Q * x(:, t) - u_j' * R * u_j + ...
                beta^2 * d_j' * d_j - 2 * u_j' * R * (ub - u_j) + ...
                2 * beta^2 * d_j' * (db - d_j));
        
        % Compute policy differences
        e1 = ub - u_j;
        e2 = db - d_j;
        
        % Accumulate state-action pairs for policy improvement
        xu = xu + dt * (kron(x(:, t), e1)');
        xd = xd + dt * (kron(x(:, t), e2)');
        
        % Update state using behavior policies
        x(:, t + 1) = x(:, t) + dt * (A * x(:, t) + B2 * ub + B1 * db);
    end

    phi_j = [phi_j; value];
    sai_j = [sai_j; [QuadraticFeatures(x(:, 1))' - QuadraticFeatures(x(:, t))', 2 * xu * kron(eye(n), R), 2 * beta^2 * xd]];
    
    L_params = sai_j / phi_j;
    P_vector = [L_params(1) L_params(2)/2 L_params(3)/2 L_params(4)/2 L_params(5)/2 L_params(6)];
    P{j+1} = VectorToSymmetricMatrix(P_vector);
    delta_P = norm(P{j+1} - P{j}, 'fro');

    K1(j+1, :) = [L_params(7) L_params(8) L_params(9)];
    K2(j+1, :) = [L_params(10) L_params(11) L_params(12)];
    
    if delta_P < 1e-4
        break;
    end
end

disp('P:');
disp(P{j+1});

disp(['K1: ', num2str(K1(j+1,:))]);

disp(['K2: ', num2str(K2(j+1,:))]);

%% Functions
function quadraticFeatures = QuadraticFeatures(vector)
    % Extract individual elements of the state vector
    s1 = vector(1); 
    s2 = vector(2); 
    s3 = vector(3);
    
    % Construct quadratic terms of the state vector
    quadraticFeatures = [s1^2, s1*s2, s1*s3, s2^2, s2*s3, s3^2]';
end

function symmetricMatrix = VectorToSymmetricMatrix(vector)
    % Reshape the vector into a symmetric matrix
    symmetricMatrix = [vector(1)    vector(2)/2  vector(3)/2;
                       vector(2)/2  vector(4)    vector(5)/2;
                       vector(3)/2  vector(5)/2  vector(6)];
end