clearvars; close all; clc;

%% Problem 1
% CompressedUtils.solveOMP
% CompressedUtils.solveSP

%% Problem 2
load('ps1_2022.mat');
A_all = {Af,Ar}; %Sensing
y_all = {yf,yr}; %Measurement
text_all = {'Af', 'Ar'};
S = 3;

% A
for i = 1:numel(A_all)
    [~, x_error] = CompressedUtils.solveOMP(A_all{i},y_all{i},S);
    fprintf("%s Error for OMP: %.4e\n", text_all{i}, x_error);
end

% B
for i = 1:numel(A_all)
    [~, x_error] = CompressedUtils.solveSP(A_all{i},y_all{i},S);
    fprintf("%s Error for SP: %.4e\n", text_all{i}, x_error);
end

% [omp_x,sp_x]

%% Problem 3
N = 256; S = 5;

tol = 10e-6;
num_meas = 0:10:100;
n_iter = 100;

count_sp = zeros(size(num_meas));
count_omp = zeros(size(num_meas));

for i = 1:numel(num_meas)
    M = num_meas(i);

    for j = 1:n_iter
        x=zeros(N,1); q=randsample(1:N,S); x(q)=randn(S,1);
        A=randn(M, N); A=orth(A')';

        y = A*x;

        x_SP = CompressedUtils.solveSP(A,y,S);
        error_SP = norm(x_SP-x, 2);

        x_OMP = CompressedUtils.solveOMP(A,y,S);
        error_OMP = norm(x_OMP-x, 2);

        count_sp(i)  = count_sp(i)  + (error_SP<tol);
        count_omp(i) = count_omp(i) + (error_OMP<tol);
    end
end
count_sp  = (count_sp./n_iter)*100;
count_omp = (count_omp./n_iter)*100;

figure;
hold on;
plot(num_meas, count_omp, "-o", "LineWidth", 2, "DisplayName", "OMP");
plot(num_meas, count_sp, "-s", "LineWidth", 2, "DisplayName", "SP");
hold off;
grid on;
xlabel("Number of Measurements (#)");
ylabel("Probability of Perfect Recovery (%)");
title(sprintf("Performance of OMP vs. SP (N=%d, S=%d)", N, S));
legend("Location", "SouthEast");
set(gca, "FontSize", 12);