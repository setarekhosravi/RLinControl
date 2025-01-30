clc;
clear;
close all;

%% Plot System Variables
A = [0.5 1.5; 2 -2];
B = [1 4]';
n = size(A, 1);

K_IRL = [1.6163     0.89853];


t = 0:0.01:10;

x0 = [5 -5]';

% Function to represent the system dynamics (x'(t) = Ax(t) + Bu(t))
system = @(t, x) (A - B * K_IRL) * x;

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