clearvars; close all; clc;

%% Problem 1
N = 256; S = 5;

tol = 10e-6;
num_meas = 0:10:100;
n_iter = 100;

count_sp = zeros(size(num_meas));
count_omp = zeros(size(num_meas));
count_l1 = zeros(size(num_meas));

for i = 1:numel(num_meas)
    M = num_meas(i);

    for j = 1:n_iter
        x=zeros(N,1); q=randsample(1:N,S); x(q)=randn(S,1);
        A=randn(M, N); A=orth(A')';

        y = A*x;

        x_SP = CompresfsedUtils.solveSP(A,y,S);
        error_SP = norm(x_SP-x, 2);

        x_OMP = CompressedUtils.solveOMP(A,y,S);
        error_OMP = norm(x_OMP-x, 2);

        x_L1 = CompressedUtils.solveL1(A,y);
        error_l1 = norm(x_L1-x, 2);

        count_sp(i)  = count_sp(i)  + (error_SP<tol);
        count_omp(i) = count_omp(i) + (error_OMP<tol);
        count_l1(i)  = count_l1(i) + (error_l1<tol);
    end
end
count_sp  = (count_sp./n_iter)*100;
count_omp = (count_omp./n_iter)*100;
count_l1  = (count_l1./n_iter)*100;


figure;
hold on;
plot(num_meas, count_omp, "-o", "LineWidth", 2, "DisplayName", "OMP");
plot(num_meas, count_sp, "-s", "LineWidth", 2, "DisplayName", "SP");
plot(num_meas, count_l1, "-s", "LineWidth", 2, "DisplayName", "L1");
hold off;
grid on;
xlabel("Number of Measurements (#)");
ylabel("Probability of Perfect Recovery (%)");
title(sprintf("Performance of OMP vs. SP vs. L1 (N=%d, S=%d)", N, S));
legend("Location", "SouthEast");
set(gca, "FontSize", 12);
