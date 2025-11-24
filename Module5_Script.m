clear; clc; close all;

%% Setup Parameters
N = 256;         % Signal length
S = 5;           % Sparsity level
M_vals = 10:5:100; % Measurements to test
num_trials = 50; % Number of trials for averaging

% Different scenarios for the experiment
signal_types = {'time_sparse', 'freq_sparse'};
matrix_types = {'a', 'b', 'c', 'd', 'e', 'f'};
algo_names = {'OMP', 'l1-minimization'};

%% Results Storage
avg_nmse = zeros(length(signal_types), length(matrix_types), length(M_vals), length(algo_names));

%% Simulation Loop
for signal_idx = 1:length(signal_types)
    signal_type = signal_types{signal_idx};

    for matrix_idx = 1:length(matrix_types)
        matrix_type = matrix_types{matrix_idx};

        for m_idx = 1:length(M_vals)
            M = M_vals(m_idx);

            trial_nmse = zeros(num_trials, length(algo_names));

            parfor trial = 1:num_trials

                I = eye(N);
                F = dct(I); % Orthonormal DCT matrix
                % Build specific type of measurement matrix
                switch matrix_type
                    case 'a' % Random sampling in the time domain
                        A = I(randperm(N,M), :);

                    case 'b' % Uniform subsampling in the time domain
                        A = I(round(linspace(1, N, M)), :);

                    case 'c' % Random sampling in the frequency domain
                        A = F(randperm(N, M), :);

                    case 'd' % Low-frequency sampling
                        A = F(1:M, :);

                    case 'e' % Equispaced frequency sampling
                        A = F(round(linspace(1, N, M)), :);

                    case 'f' % Sampling in a random domain (Gaussian)
                        A = orth(randn(M, N)')'; % Orthonormalize the rows

                    otherwise
                        error('Unknown matrix type specified.');
                end

                % Build specific type of signal
                x = zeros(N, 1);
                x(randperm(N,S)) = randn(S, 1);

                switch signal_type
                    case 'time_sparse'
                        % just continue
                    case 'freq_sparse'
                        x = idct(x);
                    otherwise
                        error('Unknown signal type specified.');
                end

                % Get measurement
                y = A * x;

                % OMP
                x_omp = CompressedUtils.solveOMP(A, y, S);
                nmse_omp = norm(x - x_omp)^2 / norm(x)^2;

                % L1
                switch signal_type
                    case "time_sparse"
                        x_l1 = CompressedUtils.solveL1(A, y);
                    case "freq_sparse"
                        F = dct(eye(N));
                        A = A * F';
                        x_l1 = CompressedUtils.solveL1(A, y);
                        x_l1 = idct(x_l1);
                    otherwise
                        error('Unknown signal type specified.');
                end
                nmse_l1 = norm(x - x_l1)^2 / norm(x)^2;

                trial_nmse(trial, :) = [nmse_omp, nmse_l1];
            end

            % Average over trials
            avg_nmse(signal_idx, matrix_idx, m_idx, :) = mean(trial_nmse, 1);
        end
    end
end


%% Plot Results
matrix_titles = {
    '(a) Random Time Sampling', ...
    '(b) Uniform Time Sampling', ...
    '(c) Random Frequency Sampling', ...
    '(d) Low-Frequency Sampling', ...
    '(e) Equispaced Frequency Sampling', ...
    '(f) Random Gaussian Matrix'
    };

for signal_idx = 1:length(signal_types)
    figure('Name', ['Signal: ' signal_types{signal_idx}], 'NumberTitle', 'off', 'Position', [100, 100, 1200, 700]);
    sgtitle(['Recovery for ' strrep(signal_types{signal_idx}, '_', ' ') ' Signal (S=5)'], ...
        'FontSize', 16, 'FontWeight', 'bold');

    for matrix_idx = 1:length(matrix_types)
        subplot(2, 3, matrix_idx);
        hold on;
        for algo_idx = 1:length(algo_names)
            plot(M_vals, squeeze(avg_nmse(signal_idx, matrix_idx, :, algo_idx)), ...
                'o-', 'LineWidth', 2, 'MarkerSize', 6);
        end
        hold off;

        title(matrix_titles{matrix_idx}, 'FontSize', 12);
        xlabel('Number of Measurements (M)');
        ylabel('Average NMSE');
        grid on;
        set(gca, 'YScale', 'log');
        ylim([1e-6, 2]);

        if matrix_idx == 3
            legend(algo_names, 'Location', 'southwest');
        end
    end
end