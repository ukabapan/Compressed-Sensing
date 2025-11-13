classdef CompressedUtils

    methods(Static)
        function [x_hat, solution_error] = solveES(A, y, S)
% Exhaustive Search algorithm for sparse signal recovery
%
% Inputs:
%   A     - Sensing matrix (M x N)
%   y     - Measurement vector (M x C)
%   S     - Sparsity level (number of non-zero elements in x)
%
% Output:
%   x_hat - Reconstructed sparse signal (N x C)
%   solution_error - Error between A*x_hat and y

arguments
    A {mustBeNumeric}
    y {mustBeNumeric}
    S {mustBeInteger, mustBePositive}
end

% Get dimension of sensing and measures
[M, N] = size(A);
[My, C] = size(y);

% Validate that a and b have the same number of rows
if M ~= My
    error('A and y must be have the same number of rows. %d',My);
end

r = y; % Initialize the residual
T = double.empty(0,S); % Initialize the support set
x_hat_T = zeros(S,C);  % Initialize the sparse signal estimate

for ss = 1:S
    K = nchoosek(1:N, ss);
    for i =  1:size(K, 1)
        % Select column subset
        T_i = K(i,:);

        % Solve the lq to get best estimate with the support set
        x_hat_T_i = pinv(A(:,T_i))*y;

        % Get residual for best estimate
        r_i = y - A(:,T_i)*x_hat_T_i;

        % If residual didn't decrease OR we were below tolerance
        if norm(r_i,2) < norm(r,2)
            % Update residual, support set, and sparse x then continue
            r = r_i;
            T = T_i;
            x_hat_T = x_hat_T_i;
        end
    end
end

% Build full signal estimate vector
x_hat = zeros(N,C);
if ~isempty(T)
    x_hat(T,:) = x_hat_T;
end


solution_error = norm(y-A*x_hat, 2);
        end

        function [x_hat, solution_error] = solveOMP(A, y, S, tol)
% Orthogonal Matching Pursuit (OMP) algorithm to recover a sparse signal.
%
% Inputs:
%   A     - Sensing matrix (M x N)
%   y     - Measurement vector (M x C)
%   S     - Sparsity level (number of non-zero elements in x)
%   tol   - Optional tolerance to stop iterating
%
% Output:
%   x_hat - Reconstructed sparse signal (N x C)
%   solution_error - Error between A*x_hat and y

arguments
    A   {mustBeNumeric}
    y   {mustBeNumeric}
    S   {mustBeInteger, mustBePositive}
    tol {mustBeReal, mustBePositive} = 1e-15
end

% Get dimension of sensing and measures
[M, N] = size(A);
[My, C] = size(y);

% Validate that a and b have the same number of rows
if M ~= My
    error('A and y must be have the same number of rows.');
end

r = y; % Initialize the residual
T = double.empty(0,S); % Initialize the support set
x_hat_T = zeros(S,C);  % Initialize the sparse signal estimate

for i = 1:S

    % Add most correlated column indice to support set
    [~, J] = max(abs(A'*r));
    T_i = union(T,J);

    % Solve the lq to get best estimate with the support set
    x_hat_T_i = pinv(A(:,T_i))*y;

    % Get residual for best estimate
    r_i = y - A(:,T_i)*x_hat_T_i;

    if  (norm(r_i,2) >= norm(r,2)) || (norm(r, 2) < tol)
        break;
    else
        % Update residual, support set, and sparse x then continue
        r = r_i;
        T = T_i;
        x_hat_T = x_hat_T_i;
    end
end

% Build full signal estimate vector
x_hat = zeros(N,C);
if ~isempty(T)
    x_hat(T,:) = x_hat_T;
end

solution_error = norm(y-A*x_hat, 2);
        end

        function [x_hat, solution_error] = solveSP(A, y, S, tol)
% Subspace Pursuit (SP) algorithm to recover a sparse signal.
%
% Inputs:
%   A     - Sensing matrix (M x N)
%   y     - Measurement vector (M x C)
%   S     - Sparsity level (number of non-zero elements in x)
%   tol   - Optional tolerance to stop iterating
%
% Output:
%   x_hat - Reconstructed sparse signal (N x C)
%   solution_error- Error between A*x_hat and y

arguments
    A   {mustBeNumeric}
    y   {mustBeNumeric}
    S   {mustBeInteger, mustBePositive}
    tol {mustBeReal, mustBePositive} = 1e-15
end

% Get dimension of sensing and measures
[M, N] = size(A);
[My, C] = size(y);

% Validate that a and b have the same number of rows
if M ~= My
    error('A and y must be have the same number of rows.');
end

r = y; % Initialize the residual
T = double.empty(0,S); % Initialize the support set
x_hat_T = zeros(S,C);  % Initialize the sparse signal estimate

for i = 1:nchoosek(N,S) % Theoretical max you should need

    % Add top S correlated column indice to support set
    [~, J] = maxk(abs(A' * r), S);
    T_i = union(T, J');

    v = pinv(A(:,T_i)) * y;

    % sigmaS operation
    [~, idx] = maxk(abs(v),S);
    v_s = zeros(size(v), 'like', v);
    v_s(idx) = v(idx);
    x_hat_T_i = v_s;

    T_i(~find(x_hat_T_i)) = [];
    x_hat_T_i(~find(x_hat_T_i)) = [];

    % Get residual for best estimate
    r_i = y - A(:,T_i)*x_hat_T_i;


    if  (norm(r_i,2) >= norm(r,2)) || (norm(r, 2) < tol)
        break;
    else

        % Update residual, support set, and sparse x then continue
        r = r_i;
        T = T_i;
        x_hat_T = x_hat_T_i;
    end
end

% Build full signal estimate vector
x_hat = zeros(N,C);
if ~isempty(T)
    x_hat(T,:) = x_hat_T;
end

solution_error = norm(y-A*x_hat, 2);
end

        function [x_hat, solution_error] = solveL1(A, y)
% L1-minimization algorithm to recover a time or frequency sparse signal.
%
% Inputs:
%   A            - Sensing matrix (M x N)
%   y            - Measurement vector (M x C)
%
% Output:
%   x_hat        - Reconstructed sparse signal (N x C)
%   solution_error       - Error between A*x_hat and y

arguments
    A {mustBeNumeric}
    y {mustBeNumeric}
end

% Get dimension of sensing and measures
[M, N] = size(A);
[My, C] = size(y);

% Validate that a and b have the same number of rows
if M ~= My
    error('A and y must be have the same number of rows A = %d, y = %d.',M, My);
end

% : min f'*z s.t. Aeq*z = beq, z >= 0
% Let x = u - v, where u,v >= 0. Then ||x||_1 = sum(u+v).
% z = [u; v]

f = ones(2*N, C); % Objective function to minimize sum(u+v)
Aeq = [A, -A];    % Constraint A(u-v) = y
beq = y;
lb = zeros(2*N, C); % Lower bound u,v >= 0

options = optimoptions('linprog', 'Display', 'none');

z = linprog(f, [], [], Aeq, beq, lb, [], options);

if isempty(z)
    x_hat = zeros(N, C);
else
    % Reconstruct x from u and v
    x_hat = z(1:N, :) - z(N+1:2*N, :);
end

solution_error = norm(y-A*x_hat, 2);
        end


    end
end