% System matrices
A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B2 = [0 0 1]';
B1 = [1 0 0]';
C = [1, 1, 1];
x0 = [10; -10; -3]; % Initial condition

% Feedback gain matrices
K1 = [-0.087019, 0.086473, 0.02603]; % From your provided data

% Input disturbance
w = @(t) 0.2 * exp(-0.2 * t) .* cos(t);

% Time span for simulation
tspan = [0, 20];
dt = 0.01; % Time step
t = tspan(1):dt:tspan(2);

% Preallocations
x = zeros(3, length(t));
x(:, 1) = x0; % Initial state
u = zeros(1, length(t));
z = zeros(1, length(t));
w_vals = w(t);
rd = zeros(1, length(t)); % Real attenuation coefficient

% Simulation loop
for i = 1:length(t)-1
    % Control input
    u(i) = -K1 * x(:, i);
    
    % State derivative
    dx = A * x(:, i) + B2 * u(i) + B1 * w(t(i));
    
    % Update state using Euler method
    x(:, i+1) = x(:, i) + dx * dt;
    
    % Output
    z(i) = C * x(:, i);
end

% Last output calculation
z(end) = C * x(:, end);

% Calculate rd(t)
numerator = cumtrapz(t, z.^2 + u.^2);
denominator = cumtrapz(t, w_vals.^2);
rd = sqrt(numerator ./ denominator);

% Plot results
figure;
plot(t, rd, 'LineWidth', 1.5);
xlabel('Time (t)');
ylabel('Real Attenuation Coefficient (r_d(t))');
title('Real Attenuation Coefficient vs Time');
grid on;
