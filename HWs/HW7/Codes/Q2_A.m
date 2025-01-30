clc;
clear;
close all;
%% P Calculation
A = [-1.01887 0.90506 -0.00215; 0.82225 -1.07741 -0.17555; 0 0 -1];
B2 = [0 0 1]';
B1 = [1 0 0]';
B = [B1 B2];

n = size(A , 1) ;

Q = eye(n);
beta = 5;

m1 = size(B1,2);
m2 = size(B2,2);
R = [-beta^2*eye(m1) zeros(m1,m2) ; zeros(m2,m1) eye(m2)];

[P_care,G,K]=care(A,B,Q,R);
disp('CARE P:');
disp(P_care);