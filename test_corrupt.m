clear; clc; close all;

fprintf('Loading data...\n');
load('AssignmentsData/CroppedYale_96_84_2414_subset.mat');

% --- 1. Data Preparation ---
% Note: faces is 2414x96x84. faces(:,:)' is the correct 8064x2414 (m x N)
all_images = double(faces(:, :)');
colmin = min(all_images);
colmax = max(all_images);
all_images = rescale(all_images,"InputMin",colmin,"InputMax",colmax);
all_labels = facecls;

rng("default");
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

% --- 2. PCA Projection (Eigenfaces) ---
fprintf('Performing PCA on training data...\n');
[A,A_mu, ~] = normalize(A, "center");
num_components = 1000;
[coeff, ~, ~, ~, explained] = pca(A', 'NumComponents', num_components, 'Centered','off');
A = coeff' * A; % Project training data (k x N_train)
fprintf('PCA complete. Projected dimension: %d\n', num_components);

% --- 3. Pre-computation for Classification ---
% This mask is (N_train x num_classes)
labels_mask = (A_labels == (1:num_classes));

% --- 4. *** SPEEDUP: Pre-compute ADMM Solver Factors *** ---
% We pre-calculate the Cholesky factorization of (A'A + rho*I)
% This is the most expensive part of the solver, and we do it only ONCE.
fprintf('Pre-computing Cholesky factorization for fast ADMM solver...\n');
rho = 1.0; % ADMM penalty parameter
n_train = size(A, 2); 
lambda_lasso = 1e-4; % Lambda from your original script
% L*L' = (A'A + rho*I)
L = chol(A'*A + rho * speye(n_train), 'lower');
L_t = L';
fprintf('Factorization complete.\n');

% --- 5. Setup Parallel Pool ---
if isempty(gcp('nocreate'))
    parpool;
end

% --- 6. Main Computation Loop ---
fprintf('Starting parallel computation...\n');
tic

% Get image dimensions (Fixed the 'sizeim' error)
IMG_HEIGHT = size(faces, 2); % 96
IMG_WIDTH  = size(faces, 3); % 84

corruption_levels = linspace(0,0.4,2); % 0% and 40%

% Pre-generate all corruption data (this loop is fast)
corruption_idx = zeros(floor(num_tests * IMG_HEIGHT * IMG_WIDTH * max(corruption_levels)), length(corruption_levels));
corruption_vals = zeros(size(corruption_idx));
disp("Generating corruption data...");
for ii = 1:length(corruption_levels)
    [idx, vals] = get_block_occlusion_data(Y, IMG_HEIGHT, IMG_WIDTH, corruption_levels(ii));
    if ~isempty(idx)
        corruption_idx(1:length(idx), ii) = idx;
        corruption_vals(1:length(vals), ii) = vals;
    end
end
disp("Corruption data generated.");

predicted_classes = zeros(num_tests,length(corruption_levels));

for cc = 1:length(corruption_levels)
    fprintf('Processing corruption level %.2f...\n', corruption_levels(cc));
    
    % Apply corruption
    Y_corrupt_orig = Y; % Start from original un-projected data
    idx_mask = corruption_idx(:,cc) > 0;
    current_idx = corruption_idx(idx_mask, cc);
    current_vals = corruption_vals(idx_mask, cc);
    
    if ~isempty(current_idx)
        Y_corrupt_orig(current_idx) = current_vals;
    end
    
    % Project the *entire* corrupted test set at once
    % Use the *training mean* (A_mu) for centering, as is correct
    Y_corrupt_pca = coeff' * (Y_corrupt_orig - A_mu); 
    
    fprintf('  Solving for %d test images in parallel...\n', num_tests);
    
    parfor ii = 1:num_tests
        % Get the projected, corrupted test vector
        y_vec = Y_corrupt_pca(:,ii);
        
        % --- *** SPEEDUP: Use custom fast solver *** ---
        % This is much faster than calling lasso()
        x_hat = solve_lasso_admm_custom(A, y_vec, lambda_lasso, rho, L, L_t);
        
        % Classification (this part was already fast and is unchanged)
        residuals_per_class = vecnorm(y_vec - A * (x_hat.*labels_mask));
        [~, predicted_classes(ii,cc)] = min(residuals_per_class);
    end
    
    fprintf('  Corruption level %.2f complete.\n', corruption_levels(cc));
end

toc
fprintf('All computations finished.\n');

% --- 7. Calculate Accuracy ---
accuracy_rate = zeros(length(corruption_levels), 1);
for cc = 1:length(corruption_levels)
    accuracy_rate(cc) = mean(predicted_classes(:,cc) == Y_labels);
    fprintf('Accuracy at %.0f%% corruption: %.2f%%\n', corruption_levels(cc)*100, accuracy_rate(cc)*100);
end


% =========================================================================
% LOCAL HELPER FUNCTIONS FOR FAST LASSO
% =========================================================================

function [x] = solve_lasso_admm_custom(A, y, lambda, rho, L, L_t)
    % Solves the LASSO problem: min 0.5*||Ax-y||_2^2 + lambda*||x||_1
    % This version uses pre-computed Cholesky factors (L, L_t) for speed.
    
    n = size(A, 2);
    MAX_ITER = 100;
    TOL = 1e-4;

    % Precompute A'y
    A_ty = A' * y;
    
    % Initialize
    x = zeros(n, 1);
    z = zeros(n, 1);
    u = zeros(n, 1); % Scaled dual variable

    for k = 1:MAX_ITER
        % x-update: (A'A + rho*I)x = A'y + rho*(z - u)
        b = A_ty + rho * (z - u);
        
        % --- FAST SOLVE ---
        % Solve using pre-computed Cholesky factors: L*L'*x = b
        x_new = L_t \ (L \ b);
        
        % z-update: z = S_lambda/rho(x + u)
        z_new = soft_thresh(x_new + u, lambda / rho);
        
        % u-update: u = u + x - z
        u_new = u + x_new - z_new;
        
        % Check for convergence (primal and dual residuals)
        if norm(x_new - z_new) < TOL && norm(rho * (z_new - z)) < TOL
            x = z_new;
            break;
        end
        x = x_new;
        z = z_new;
        u = u_new;
    end
    x = z; % The sparse solution
end

function [z] = soft_thresh(x, t)
    % Soft-thresholding operator
    z = sign(x) .* max(abs(x) - t, 0);
end

function [all_indices, all_values] = get_block_occlusion_data(Y, IMG_HEIGHT, IMG_WIDTH, occlusion_percentage)
    
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