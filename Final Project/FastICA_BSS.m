clear; clc; close all;

%% Data Loading
audio_file_location = "audio/synthetic_audio.wav";
truncate_duration = 0; % Set to 0 for no truncation

[X,Fs] = load_stereo_audio(audio_file_location,truncate_duration);
fprintf('Data loaded. Samples: %d\n', length(X));

%% Pre-processing
% Centering
[N, M] = size(X);
mean_X = mean(X, 2);
X_centered = X - mean_X;

% Whitening
Cov_X = (X_centered * X_centered') / (M - 1);
[E, D] = eig(Cov_X);
V = sqrt(D)\(E');
Z = V * X_centered;

fprintf('Pre-processing complete. Data centered and whitened.\n');

%% FastICA Algorithm implementation
% Number of independent components to extract
num_IC = N;

% Initialize demixing matrix W with random values
W = randn(num_IC, N);

% Parameters for Newton iteration
max_iter = 1000;
tol = 1e-5;

fprintf('Starting FastICA iterations...\n');

for i = 1:num_IC
    % Vector w for the i-th component
    w = W(i, :)';
    w = w / norm(w); % Normalize

    for iter = 1:max_iter
        w_old = w;

        % Hyperbolic tangent is a robust choice for the non-linearity g(u)
        u = w' * Z;
        g_u = tanh(u);
        g_prime_u = 1 - g_u.^2;

        % FastICA Update Rule:
        % w+ = E[Z * g(w'Z)] - E[g'(w'Z)] * w
        term1 = mean(Z .* g_u, 2);
        term2 = mean(g_prime_u) * w;
        w = term1 - term2;

        % Deflation / Decorrelation:
        % Subtract projections onto previously found vectors to ensure independence
        if i > 1
            w = w - W(1:i-1, :)' * (W(1:i-1, :) * w);
        end

        % Normalize
        w = w / norm(w);

        % Check convergence (direction matters, not sign)
        if abs(abs(w' * w_old) - 1) < tol
            fprintf('Component %d converged in %d iterations.\n', i, iter);
            break;
        end
    end
    W(i, :) = w';
end

%% Source Reconstruction
% The estimated independent components
Y = W * Z;

%% Visualization
figure('Name', 'Blind Source Separation Results', 'Color', 'white');

% Plot Original Mixes
subplot(3, 2, 1); plot(X(1, 1:500)); title('Observed Mix Channel 1'); grid on;
subplot(3, 2, 2); plot(X(2, 1:500)); title('Observed Mix Channel 2'); grid on;

% Plot Separated Signals
subplot(3, 2, 3); plot(Y(1, 1:500)); title('Recovered Component 1'); grid on;
subplot(3, 2, 4); plot(Y(2, 1:500)); title('Recovered Component 2'); grid on;

% Joint distribution plot (Scatter)
subplot(3, 2, 5);
scatter(X(1, 1:1000), X(2, 1:1000), 5, 'filled');
title('Joint Plot: Mixed Signals');
xlabel('Mic 1'); ylabel('Mic 2'); axis square;

subplot(3, 2, 6);
scatter(Y(1, 1:1000), Y(2, 1:1000), 5, 'filled');
title('Joint Plot: Recovered Signals');
xlabel('Comp 1'); ylabel('Comp 2'); axis square;

%% Save Output Audio
% Normalize amplitude to avoid clipping when saving
Y_norm = Y ./ max(abs(Y), [], 2);

filename1 = 'output_source_1.wav';
filename2 = 'output_source_2.wav';
audiowrite(filename1, Y_norm(1, :), Fs);
audiowrite(filename2, Y_norm(2, :), Fs);
fprintf('Separated audio saved to %s and %s\n', filename1, filename2);
