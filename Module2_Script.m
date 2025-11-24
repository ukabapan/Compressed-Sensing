clearvars; close all; clc;
load('ps1_2022.mat');

A_all = {Af, Ar};
y_all = {yf, yr};
txt_all = {'Af', 'Ar'};
S = 3;

%% Problem 1
for i = 1:numel(A_all)
    [~, error] = CompressedUtils.solveES(A_all{i}, y_all{i}, S);
    fprintf("Error for %s: %.4e\n", txt_all{i}, error)
end

%% Problem 2
% Most of the elements of the Af matrix are mirrored with the exception of
% sign changes here and there.
% Ar array seems normal otherwise
% The error for the Af matrix ends up being much lower as a result of its
% redundancies