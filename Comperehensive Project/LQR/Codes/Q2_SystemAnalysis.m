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
x0 = [5 -5]';

disp("Eigen Values of A:")
disp(eig(A))

disp("A rank:")
disp(rank(ctrb(A,B)))
%% Design admissible policy
desired_poles = [-2 -4];
K = place(A,B,desired_poles);

disp("Stabilizing K: ")
disp(K)
%% Final analysis
disp("Eigen Values of New System:")
disp(eig(A-B*K))