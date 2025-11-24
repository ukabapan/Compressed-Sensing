clear; clc; close all;

%% Setup
image_files = {'phantom.png', 'brain256.png', 'Boat.tif'};
image_names = {'Phantom', 'Brain', 'Boat'};
sensing_types = {'Gaussian', 'Subsampling', 'SRM'};
patch_size = 16;
M_fractions = 0.1:0.1:0.5; % Fraction of measurements

N_patch = patch_size * patch_size;
M_values = round(M_fractions * N_patch); % Absolute number of measurements

num_images = length(image_files);
num_sensing = length(sensing_types);
num_m_values = length(M_values);

psnr_results = zeros(num_images, num_sensing, num_m_values);


Psi = dct2dbasis(patch_size); % Synthesis matrix

%% Main Loop
MAX_INTENSITY = 255.0; % Max intensity
for i = 1:num_images

    % Load and preprocess image
    img_orig_color = imread(image_files{i});
    if size(img_orig_color, 3) == 3
        img_orig_gray = rgb2gray(img_orig_color);
    else
        img_orig_gray = img_orig_color;
    end
    % Use double precision in the original range [0, 255]
    img_orig = double(img_orig_gray);

    for s = 1:num_sensing
        sensing_type = sensing_types{s};
        for m_idx = 1:num_m_values
            M = M_values(m_idx);



            [rows, cols] = size(img_orig);


            patches_vec = im2col(img_orig, [patch_size patch_size], 'distinct');
            num_patches = size(patches_vec, 2);
            recovered_patches_vec = zeros(size(patches_vec));

            F_srm = [];

            switch sensing_type
                case 'SRM'
                    dct_1d = dctmtx(patch_size);
                    F_srm = kron(dct_1d, dct_1d);
                otherwise
            end

            parfor p = 1:num_patches
                patch_col = patches_vec(:, p);

                switch sensing_type

                    case 'Gaussian'
                        A = randn(M, N_patch);

                        A = orth(A')'; % Orthonormalize rows
                        y = A * patch_col;
                        A_eff = A * Psi;
                    case 'Subsampling'
                        A = eye(N_patch);
                        A = A(randperm(N_patch, M), :); % Select M random rows
                        y = A * patch_col;
                        A_eff = A * Psi;
                    case 'SRM'

                        D_diag = (2 * (rand(N_patch, 1) > 0.5) - 1);
                        Dx = D_diag .* patch_col;


                        FDx = F_srm * Dx;


                        row_indices = randperm(N_patch, M);
                        y = FDx(row_indices, :);

                        R_F = F_srm(row_indices, :);
                        D_Psi = diag(D_diag) * Psi;
                        A_eff = R_F * D_Psi;
                    otherwise
                        error('Unknown sensing type');
                end



                % L1 Recovery
                alpha_hat = CompressedUtils.solveL1(A_eff,y)

                if isempty(alpha_hat) || any(isnan(alpha_hat))
                    recovered_patches_vec(:, p) = zeros(N_patch, 1); % Assign zero patch if L1 fails
                else
                    recovered_patches_vec(:, p) = Psi * alpha_hat;
                end
            end

            % Reassemble image using col2im
            img_recov = col2im(recovered_patches_vec, [patch_size patch_size], [rows cols], 'distinct');

            % Clip values to be in the valid range
            img_recov = clip(img_recov,0,255);

            % Calculate MSE
            mse = mean((img_orig(:) - img_recov(:)).^2);

            % Calculate PSNR
            if mse == 0
                psnr_val = Inf; % Perfect reconstruction
            else
                psnr_val = 10 * log10(MAX_INTENSITY^2 / mse);
            end

            psnr_results(i, s, m_idx) = psnr_val;

            % Display recovered image for M value
            if m_idx == 2
                figure;
                imshow(uint8(img_recov));
                title(sprintf('%s, %s, M=%d (%.0f%%), PSNR=%.2f dB', ...
                    image_names{i}, sensing_types{s}, M, M_fractions(m_idx)*100, psnr_val));
            end

        end
    end
end

%% Plot Final Results
markers = {'-o', '-s', '-^', '-d'};

figure('Position', [100, 100, 1200, 500]);

for i = 1:num_images
    subplot(1, num_images, i);
    hold on;
    for s = 1:num_sensing
        plot(M_fractions * 100, squeeze(psnr_results(i, s, :)), ...
            markers{s}, 'LineWidth', 1.5, 'MarkerSize', 6);
    end
    hold off;
    grid on;
    xlabel('Measurement Percentage (%)');
    ylabel('PSNR (dB)');
    title(sprintf('%s (Patch Size %dx%d)', image_names{i}, patch_size, patch_size));
    if i == num_images
        legend(sensing_types, 'Location', 'SouthEast');
    end
    ylim([10, 45]); % Adjust Y-axis limits if needed
end
sgtitle('Compressed Sensing Image Recovery Performance', 'FontSize', 14, 'FontWeight', 'bold');


function C2 = dct2dbasis(Size)
% Function to compute 2D separable DCT basis functions
% Stack basis functions into rows

C1 = dctmtx(Size);
M = zeros(Size,Size);  % Preallocate matrix
for i = 1:Size
    for j = 1:Size
        M = C1(i,:)' * C1(j,:);
        C2((i-1)*Size+j,:)=M(:);
        %subplot(Size,Size,((i-1)*Size) + j), imshow(M);
    end
end
end