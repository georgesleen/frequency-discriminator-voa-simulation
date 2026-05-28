clc
close all
clear

%% Parameters

% laser E-field and frequency
EL = .1;
lambda = 1550e-9;
f0 = 300e6 / lambda;  % 1550 nm laser
w0 = 2 * pi * f0;

% bar and cross coupling of MMI
kb = sqrt(0.52);
kc = sqrt(1 - kb^2);

% delay time
syms tau_d positive

% attenuation
alpha = .98;

% responsitivity
Rave = 1.1;  % amp/watt
CMRR = 1.1/0.04;
R1 = Rave * (1 + 1 / CMRR);
R2 = Rave * (1 - 1 / CMRR);

% sampling frequency and time
fs = 10e9;
tend = 1e-4;
time = 0:1/fs:tend;
t = 0:1/fs:tend;

% amplitude noise of the laser
RIN = 10^(-150 / 10) * fs;  % S_RIN = -150 dB/Hz
rng(0);
eps_mat = sqrt(RIN) * randn(1, length(time));  % sig_eps = sqrt(RIN)
eps = @(t) eps_mat(find(time == t(1)):find(time == t(end)));

% frequency and phase noise of the laser
LW = 1e3;  % linewidth = 10 kHz
S_v = LW / pi;
rng(0);
vn_mat = sqrt(S_v * fs) * randn(1, length(time));  % sig_v = sqrt(S_v * fs)
vn = @(t) vn_mat(find(time == t(1)):find(time == t(end)));
phi_mat = cumtrapz(time, vn_mat);
phi = @(t) phi_mat(find(time == t(1)):find(time == t(end)));

% Define other parameters
K0 = -2 * (R1 + R2) * kb * kc * alpha;
K1 = R1 * kb^2 - R2 * kc^2;
K2 = alpha^2 * (R1 * kc^2 - R2 * kb^2);
E0 = kb * EL;
w = @(f) 2 * pi * f;

%% FN Dominance in System

% thermal noise parameters
R_tia = 1e6;        % TIA resistor
K_bolt = 1.38e-23;  % boltzman constant in J/K
T_k = 300;          % temperature in K
BW_n = fs;          % noise bandwidth

% PD noise parameters
q_c = 1.6e-19;  % electron charge in coulombs
I_d = 37e-9;    % dark current

% TIA thermal noise
sig_nR = 4 * K_bolt * T_k * BW_n / R_tia;  % noise variance of TIA

% PD noise (shot noise)
I_pin = 2 * I_d + R1 * E0^2 * (kb^2 + alpha^2 * kc^2) + R2 * E0^2 * (kc^2 + alpha^2 * kb^2);  % average current for the two PDs
sig_nPD = 2 * q_c * I_pin * BW_n;  % two currents were summed up for this

sig_tot = sig_nPD + sig_nR;  % total noise variance

% the desired readout
dI_pin = K0 * E0^2 * tau_d * sqrt(S_v * BW_n);  % dphi = tau * dvn where dvn = sigma_vn

tau = 2 * double(solve(dI_pin^2 / sig_tot == 10, tau_d));

%% FN Dominance in MZI
% the constant values in time (delta in frequency) domain are ignored

% heater phase shift
Dtheta = pi/2 - tau * w0;

% External PSD terms
[S_eps, fr] = PSD_auto(eps(t), fs);
S_eps = @(f) S_eps(find(fr == f(1)):find(fr == f(end)));  % PSD of epsilon

[S_vn, fr] = PSD_auto(vn(t), fs);
S_vn = @(f) S_vn(find(fr == f(1)):find(fr == f(end)));  % PSD of vn

[S_phi, fr] = PSD_auto(phi(t), fs);
S_phi = @(f) S_phi(find(fr == f(1)):find(fr == f(end)));  % PSD of phi

[S_eps_phi, fr] = PSD_cross(eps(t), phi(t), fs);
S_eps_phi = @(f) S_eps_phi(find(fr == f(1)):find(fr == f(end)));  % cross-PSD

freq = fr;

% Compute components
C_fac = cos(w0 * tau + Dtheta);
S_fac = sin(w0 * tau + Dtheta);  % the negative sign is applied in the equations

% Second term: PSD of epsilon
Term2 = @(f) E0^4 * (4 * abs(K1 * exp(1j * w(f) * tau / 2) + ...
             K2 * exp(-1j * w(f) * tau / 2) + ...
             K0 * C_fac * cos(w(f) * tau / 2)).^2 .* S_eps(f));

% Third term: PSD of phi
Term_main = @(f) E0^4 * (4 * (tau/2)^2 * K0^2 * S_fac^2 * sin(w(f) * tau / 2).^2 ./ ...
            ((w(f) * tau / 2).^2) .* S_vn(f));

% Fourth term: Cross-term
CrossTerm = @(f) E0^4 * (-8 * K0 * S_fac * sin(w(f) * tau / 2) .* ...
            (K0 * C_fac * cos(w(f) * tau / 2) .* imag(S_eps_phi(f)) + ...
            real(K1 * exp(1j * w(f) * tau / 2) + K2 * exp(-1j * w(f) * tau / 2)) .* imag(S_eps_phi(f)) + ...
            imag(K1 * exp(1j * w(f) * tau / 2) + K2 * exp(-1j * w(f) * tau / 2)) .* real(S_eps_phi(f))));

Sbpd = @(f) Term2(f) + Term_main(f) + CrossTerm(f);

%% Figures

figure
semilogx(freq, 10 * log10(abs(Sbpd(freq))), 'LineWidth', 1.5, 'Color', [.466 .674 .188 .5]);

%%
hold on
semilogx(freq, 10 * log10(Term_main(freq)), 'LineWidth', 1.5, 'Color', [.635 .078 .184 .5]);

%%
hold on
semilogx(freq, 10 * log10(abs(CrossTerm(freq))), 'LineWidth', 1.5, 'Color', [0 .447 .741 .5]);

%%
hold on
semilogx(freq, 10 * log10(Term2(freq)), 'LineWidth', 1.5, 'Color', [.929 .694 .125 .5]);
title('Power Spectral Density (PSD)');
xlabel('Frequency (Hz)');
ylabel('PSD');
grid on;
legend('total PSD', 'frequency noise', 'cross-correlation noise', 'amplitude noise', 'Location', 'southwest')

%% Delay Line

tau_delay = tau;

% delay line length
n_si = 3.48;
c = 3e8;
length = c / n_si * tau_delay;

% area occupation
width_wg = 3e-6;
gap_wg = 4.5e-6;
area = length * (width_wg + gap_wg);

% delay line loss
alpha_l = 0.2 * 1e2;  % loss = 0.2 dB/cm
loss = alpha_l * length;  % in dB

disp(['Delay time: ', num2str(tau_delay * 1e9), ' ns'])
disp(['Delay line length: ', num2str(length * 1e2), ' cm'])
disp(['Delay line area: ', num2str(area * 1e6), ' mm^2'])
disp(['Delay line loss: ', num2str(loss), ' dB'])

%% PSD Functions

function [Sx, f] = PSD_auto(x, fs)
% Power spectral density via FFT (one-sided, scaled to physical units)

    L = 2^(nextpow2(length(x)));
    X = fft(x, L);
    Sx = X .* conj(X) / L / fs;
    Sx = Sx(1:L/2);
    Sx(2:end) = 2 * Sx(2:end);
    f = linspace(0, fs/2, L/2);

end

function [Sxy, f] = PSD_cross(x, y, fs)
% Cross power spectral density via FFT (one-sided, scaled to physical units)

    L = 2^(nextpow2(length(x)));
    X = fft(x, L);
    Y = fft(y, L);
    Sxy = X .* conj(Y) / L / fs;
    Sxy = Sxy(1:L/2);
    Sxy(2:end) = 2 * Sxy(2:end);
    f = linspace(0, fs/2, L/2);

end
