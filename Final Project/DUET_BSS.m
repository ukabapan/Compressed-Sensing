clear; clc; close all;

%% 1. Load Real Audio
% Replace with your actual path
 
audio_file_location = 'audio/voice_recording.wav';

window_len = 4096;  % Long window for better FREQ resolution
smooth_amount = 1.5; % Heavy blurring to handle Messiness of real audio
N_sources = 2;

% window_len = 1024;  % Short window for better TIME resolution
% smooth_amount = 0.5; % Less blurring for sharp synthetic peaks
% N_sources = 2;

search_range_db = 20; % Standard range

[X, Fs] = load_stereo_audio(audio_file_location);

x1 = X(1,:); % Left Channel
x2 = X(2,:); % Right Channel

fprintf('Audio loaded. Fs = %d Hz.\n', Fs);

%% STFT Analysis
nfft = window_len;
hop_len = window_len / 4;
win = hamming(window_len);
[X1, f, ~] = stft(x1, Fs, 'Window', win, 'OverlapLength', window_len-hop_len, 'FFTLength', nfft);
[X2, ~, ~] = stft(x2, Fs, 'Window', win, 'OverlapLength', window_len-hop_len, 'FFTLength', nfft);

% Frequency grid for phase calculation
freq_grid = repmat(f, 1, size(X1, 2));
omega = 2 * pi * freq_grid;

%% Feature Extraction
% Calculate spatial features for every pixel in the spectrogram
R = (X2 + eps) ./ (X1 + eps);


% 0 dB = Center. Positive = Right. Negative = Left.
alpha_db = 20*log10(abs(R)); 

% Delay calculation
delta = -angle(R) ./ (omega + eps); 

% Energy Thresholding
energy = abs(X1).^2 + abs(X2).^2;
% Increased threshold slightly to ignore reverb tails
threshold = 0.4* max(energy(:));  
mask_active = energy > threshold;

vec_alpha = alpha_db(mask_active);
vec_delta = delta(mask_active);

%% Smoothed 2D Histogram
%  -20 dB to +20 dB (Captures Hard Left/Right strings)
a_edges = linspace(-search_range_db, search_range_db, 200);   % Panning (dB)

d_edges = linspace(-4e-4, 4e-4, 200); % Delay (seconds)

H = histcounts2(vec_alpha, vec_delta, a_edges, d_edges);

% Smooth the histogram to merge split peaks
H_smooth = imgaussfilt(H, smooth_amount); 

figure('Color', 'white');
imagesc(a_edges, d_edges, H_smooth'); 
set(gca, 'YDir', 'normal');
colorbar;
xlabel('Panning (dB) [-20=Left, 0=Center, +20=Right]');
ylabel('Delay (seconds)');
title('Spatial Histogram (Bright Spots = Sources)');
hold on;

%% Clustering
centers = zeros(N_sources, 2); 
temp_H = H_smooth;

fprintf('Attempting to find %d Sources...\n', N_sources);

for k = 1:N_sources
    [~, idx] = max(temp_H(:));
    [row, col] = ind2sub(size(temp_H), idx);
    
    centers(k, 1) = a_edges(row); % Alpha (dB)
    centers(k, 2) = d_edges(col); % Delta
    
    % Inhibition: Zero out the area around the peak so we find the NEXT source
    % Increased radius to prevent finding the same "fat" peak twice
    r_radius = 15; 
    r_min = max(1, row - r_radius); r_max = min(size(temp_H,1), row + r_radius);
    c_min = max(1, col - r_radius); c_max = min(size(temp_H,2), col + r_radius);
    temp_H(r_min:r_max, c_min:c_max) = 0;
    
    plot(centers(k,1), centers(k,2), 'rx', 'MarkerSize', 15, 'LineWidth', 2);
    text(centers(k,1), centers(k,2), sprintf(' S%d', k), 'Color', 'white', 'FontWeight', 'bold');
end

%% Soft Masking & Reconstruction
fprintf('Separating Sources...\n');

masks = zeros(size(X1,1), size(X1,2), N_sources);
sigma_a = 2.0; % Spread for Panning (dB)
sigma_d = 1e-4; % Spread for Delay

for k = 1:N_sources
    dist_sq = ((alpha_db - centers(k,1)) / sigma_a).^2 + ...
              ((delta    - centers(k,2)) / sigma_d).^2;
    
    masks(:,:,k) = exp(-0.5 * dist_sq);
end

% Normalize masks (Wiener)
sum_masks = sum(masks, 3) + eps;
for k = 1:N_sources
    masks(:,:,k) = masks(:,:,k) ./ sum_masks;
end

% Reconstruct
for k = 1:N_sources
    % Apply Mask
    Y1 = X1 .* masks(:,:,k);
    Y2 = X2 .* masks(:,:,k);

    % ISTFT
    y1 = istft(Y1, Fs, 'Window', win, 'OverlapLength', window_len-hop_len, 'FFTLength', nfft);
    y2 = istft(Y2, Fs, 'Window', win, 'OverlapLength', window_len-hop_len, 'FFTLength', nfft);

    %% Save Output Audio 
    sig = [y1(:), y2(:)];
    sig = sig / max(abs(sig(:))); % Normalize

    filename = sprintf('output_source_%d.wav',k);
    audiowrite(filename, sig, Fs);
    fprintf('Saved "%s" \n', filename);
end
fprintf('Done.\n');