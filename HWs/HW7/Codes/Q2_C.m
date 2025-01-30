clc;
clear;
close all;

%% Plot System Variables
A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B = [0; 0; 1];
n = size(A, 1);

K_1 = [-0.087052    0.086505    0.026037];


t = 0:0.01:10;

x0 = [10 -10 -3]';

% Function to represent the system dynamics (x'(t) = Ax(t) + Bu(t))
system = @(t, x) (A - B * K_1) * x;

% Solve the system of ODEs using ode45
[t, x] = ode45(system, t, x0);

% Plot the state response
figure;
plot(t, x);
title('State Response Over Time');
xlabel('Time (seconds)');
ylabel('State Variables');
legend('x1', 'x2', 'x3');
grid on;