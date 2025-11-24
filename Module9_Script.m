clear; clc; close all;

disp('Loading data...');
load('AssignmentsData/CroppedYale_96_84_2414_subset.mat');


all_images = double(faces(:, :)');
colmin = min(all_images);
colmax = max(all_images);
all_images = rescale(all_images,'InputMin',colmin,'InputMax',colmax);
all_labels = facecls;

rng('default');
data_partition = cvpartition(all_labels,Holdout=0.5);
train_mask = training(data_partition);
test_mask = test(data_partition);

num_classes = length(unique(all_labels));
num_tests = data_partition.TestSize;
num_train = data_partition.TrainSize;

A = all_images(:,train_mask);
Y = all_images(:, test_mask);
A_labels = all_labels(train_mask);
Y_labels = all_labels(test_mask);


fprintf('Performing PCA on training data...');
[A,A_mu, ~] = normalize(A, 'center');
num_components = 1000;
[coeff, ~, ~, ~, explained] = pca(A', 'NumComponents', num_components, 'Centered','off');
A = coeff' * A; % Project training data (k x N_train)
fprintf('Done. Projected dimension: %d\n', num_components);


labels_mask = (A_labels == (1:num_classes));


fprintf('Pre-computing Cholesky factorization for fast ADMM solver...');
rho = 1.0; % ADMM penalty parameter
n_train = size(A, 2);
lambda_lasso = 1e-4;
L = chol(A'*A + rho * speye(n_train), 'lower');
L_t = L';
disp('Done');


if isempty(gcp('nocreate'))
    parpool;
end


disp('Starting parallel computation godwilling...');
tic

% Get image dimensions
IMG_HEIGHT = size(faces, 2);
IMG_WIDTH  = size(faces, 3);

corruption_levels = linspace(0,0.6,20);

% Pre-generate all corruption data
corruption_idx = zeros(floor(num_tests * IMG_HEIGHT * IMG_WIDTH * max(corruption_levels)), length(corruption_levels));
corruption_vals = zeros(size(corruption_idx));
fprintf('Generating corruption data...');
for ii = 1:length(corruption_levels)
    [idx, vals] = get_corruption_data(Y, IMG_HEIGHT, IMG_WIDTH, corruption_levels(ii));
    if ~isempty(idx)
        corruption_idx(1:length(idx), ii) = idx;
        corruption_vals(1:length(vals), ii) = vals;
    end
end
disp('Done.');


predicted_classes = zeros(num_tests,length(corruption_levels));
for cc = 1:length(corruption_levels)
    fprintf('Processing corruption level %.2f%%...\n', corruption_levels(cc)*100);
    Y_corrupt_orig = Y;
    idx_mask = corruption_idx(:,cc) > 0;
    current_idx = corruption_idx(idx_mask, cc);
    current_vals = corruption_vals(idx_mask, cc);

    if ~isempty(current_idx)
        Y_corrupt_orig(current_idx) = current_vals;
    end
    Y_corrupt_pca = coeff' * (Y_corrupt_orig - A_mu);

    fprintf('Solving for %d sparse solutions...', num_tests);


    x_hat_matrix = solve_lasso_admm_batch(A, Y_corrupt_pca, lambda_lasso, rho, L, L_t);
    disp('Done')
    fprintf('Classifying results...');
    residuals = zeros(num_classes, num_tests);
    for ii = 1:num_tests
        % Get the solution for this test image
        x_hat = x_hat_matrix(:, ii);

        % Get the corresponding test vector
        y_vec = Y_corrupt_pca(:, ii);

        % Classification
        residuals(:, ii) = vecnorm(y_vec - A * (x_hat.*labels_mask));
    end
    [~, predicted_classes_for_cc] = min(residuals, [], 1);
    predicted_classes(:, cc) = predicted_classes_for_cc';
    disp('Done');
end


accuracy_rate = zeros(length(corruption_levels), 1);
for cc = 1:length(corruption_levels)
    accuracy_rate(cc) = mean(predicted_classes(:,cc) == Y_labels);
    fprintf('Accuracy at %.0f%% corruption: %.2f%%\n', corruption_levels(cc)*100, accuracy_rate(cc)*100);
end

plot(corruption_levels*100,accuracy_rate*100, '-o', 'LineWidth', 1.5, 'MarkerSize', 6)
grid on;
xlabel('Percentage Corruption (%)'); ylabel('Accuracy (%)');
title('Demonstration of SRC Robustness');

function [X] = solve_lasso_admm_batch(A, Y, lambda, rho, L, L_t)
% Solves the LASSO problem for an entire batch of y-vectors

[n, n_test] = deal(size(A, 2), size(Y, 2));
MAX_ITER = 100;
TOL = 1e-4;

A_tY = A' * Y;

X = zeros(n, n_test);
Z = zeros(n, n_test);
U = zeros(n, n_test);

for k = 1:MAX_ITER
    B = A_tY + rho * (Z - U);


    X_new = L_t \ (L \ B);


    Z_new = sign(X_new + U) .* max(abs(X_new + U) - lambda / rho, 0);

    U_new = U + X_new - Z_new;

    primal_res = max(abs(X_new - Z_new), [], 'all');
    dual_res = max(abs(rho * (Z_new - Z)), [], 'all');

    if primal_res < TOL && dual_res < TOL
        X = Z_new;
        break;
    end
    X = X_new;
    Z = Z_new;
    U = U_new;
end
X = Z; % The sparse solution matrix
end


function [all_indices, all_values] = get_corruption_data(Y, IMG_HEIGHT, IMG_WIDTH, occlusion_percentage)

[NUM_PIXELS, N_test] = size(Y);

if NUM_PIXELS ~= (IMG_HEIGHT * IMG_WIDTH)
    error('Image dimensions (IMG_HEIGHT * IMG_WIDTH) do not match vector size (NUM_PIXELS).');
end

num_occluded_pixels = floor(NUM_PIXELS * occlusion_percentage);
block_size = floor(sqrt(num_occluded_pixels));

if block_size < 1
    all_indices = [];
    all_values = [];
    return;
end

max_row_start = IMG_HEIGHT - block_size + 1;
max_col_start = IMG_WIDTH - block_size + 1;


cell_indices = cell(1, N_test);
cell_values = cell(1, N_test);

parfor i = 1:N_test
    row_start = randi(max_row_start);
    col_start = randi(max_col_start);
    row_end = row_start + block_size - 1;
    col_end = col_start + block_size - 1;

    occlusion_block_values = rand(block_size, block_size);

    [cols_in_block, rows_in_block] = meshgrid(col_start:col_end, row_start:row_end);

    linear_indices_2d = sub2ind([IMG_HEIGHT, IMG_WIDTH], rows_in_block(:), cols_in_block(:));

    linear_indices_Y = sub2ind(size(Y), linear_indices_2d, repmat(i, length(linear_indices_2d), 1));

    cell_indices{i} = linear_indices_Y;
    cell_values{i} = occlusion_block_values(:);
end

all_indices = cat(1, cell_indices{:});
all_values = cat(1, cell_values{:});

end