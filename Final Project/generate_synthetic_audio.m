clear; clc; close all;

%% Data Generation
Fs = 8000;
T = 10;
t = 0:1/Fs:T-1/Fs;

% Sawtooth wave (simulates instrumental/buzz)
s1 = sawtooth(2*pi*50*t);

% Amplitude Modulated Noise (simulates percussion/speech)
s2 = randn(size(t)) .* sin(2*pi*2*t);

% Combine into a source matrix
S_true = [s1; s2];

% Create a Mixing Matrix (Linear combination)
% This simulates the "recording" environment
A_true = [0.8, 0.3;   % Mic 1 hears mostly S1
          0.4, 0.7];  % Mic 2 hears mix of both

% Generate the Observed Signal (Mix)
X = A_true * S_true;

X_norm = X ./ max(abs(X), [], 2);
audiowrite("synthetic_audio.wav", X_norm', Fs);
fprintf('Audio generated. Samples: %d\n', length(X));
