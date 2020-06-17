clear; clc;
addpath('/Users/anirudh/Desktop/FYP/sap-voicebox/voicebox');
%% Read files from text file 

training_size = 40;

aem_fileID = fopen('aem.txt','r');
aem_cell = textscan(aem_fileID,'%s','Delimiter','\n');


iem_fileID = fopen('iem.txt','r');
iem_cell = textscan(iem_fileID,'%s','Delimiter','\n');

aem_cell = aem_cell{1,1};
iem_cell = iem_cell{1,1};

%% STFT parameters
n_fft = 2048;
win_length = 400;
hop_length = 0.4*win_length;
smoothing_win_length = 15;

%% Train Transfer function 

for i=1:training_size
    
    aem_filepath = aem_cell{i,1};
    iem_filepath = iem_cell{i,1};
    
    % load AEM and IEM audio files
    [x_aem, fs] = audioread(aem_filepath); 
    [x_iem, fs] = audioread(iem_filepath); 
    
    %normalises
    Fs_new = 16000;
    [x_aem, x_iem, ~] = preproc(x_aem, x_iem, fs, 16000);
    
    % STFT
     [s_aem, ~, ~, P_aem] = spectrogram(x_aem, win_length, hop_length, n_fft, Fs_new, 'yaxis');
     [s_iem, F, T, P_iem] = spectrogram(x_iem, win_length, hop_length, n_fft, Fs_new, 'yaxis');

    % Equalisation
    s_aem = abs(s_aem);
    s_iem = abs(s_iem) ;
    
    h_eq = s_aem./s_iem;
    h_eq_avg = mean(h_eq,2);

    %apply smoothing window 
    h_eq_avg_smooth = smoothdata(h_eq_avg, 'movmean', smoothing_win_length);
    
    % matrix containing equalisation gain for each audio file pair
    h_eq_mat(:,i) = h_eq_avg;
    h_eq_mat_smooth(:,i) = h_eq_avg_smooth;
    
    % estimated transfer function
    h = mean(h_eq_mat,2);
    h_smooth = mean(h_eq_mat_smooth,2);
end
disp('training done')

%% frequency response of estimated transfer function
figure(4)
subplot(1,2,1)
% set(gca, 'FontSize', 12)
plot(F,pow2db(h), 'linewidth', 1.5);
xlabel('\fontsize{14}Frequency (Hz)')
ylabel('\fontsize{14} Spectral Gain (dB)')
title('\fontsize{16} Equalizer Function')
xlim([300,8000])


subplot(1,2,2)
set(gca, 'FontSize', 12)
plot(F,pow2db(h_smooth), 'linewidth', 1.5);
xlabel('\fontsize{14} Frequency (Hz)')
ylabel('\fontsize{14} Spectral Gain (dB)')
title('\fontsize{16} Moving-Average Equalizer Function')
xlim([300,8000])

%% Evaluate estimated transfer function
[x_test_aem, fs] = audioread('/Users/anirudh/Desktop/FYP/audio_files/aem/sp1_41_aem.flac'); 
[x_test_iem, fs] = audioread('/Users/anirudh/Desktop/FYP/audio_files/iem/sp1_41_iem.flac'); 

[x_test_aem, x_test_iem, F] = preproc(x_test_aem, x_test_iem, fs, 16000);

[s_test_aem, ~, ~, P_test_aem] = spectrogram(x_test_aem, win_length, hop_length, n_fft, Fs_new, 'yaxis');
[s_test_iem, F, T, P_test_iem] = spectrogram(x_test_iem, win_length, hop_length, n_fft, Fs_new, 'yaxis');

% apply learnt transfer function to the test set
est_s_test_aem = h_smooth .* abs(s_test_iem);
est_P_aem = pow2db(est_s_test_aem);

%% spectrogram plot
figure(1)

subplot(3,1,1)
surf(T,F,pow2db(abs(est_s_test_aem).^2), "LineStyle", "none")
view(2) 
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
xlim([T(1) T(end)])
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Estimated AEM speech')
% 
subplot(3,1,2)
hold on
box on
surf(T,F,pow2db(abs(s_test_aem)),'edgecolor', 'none');
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
xlim([T(1) T(end)])
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('External Microphone(AEM)')

subplot(3,1,3)
hold on
box on
surf(T,F,pow2db(abs(s_test_iem)),'edgecolor', 'none');
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
xlim([T(1) T(end)])
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Inner Ear Microphone(IEM)')

% PSD
LSD=LogSpectralDistance(x_test_aem,x_test_iem,16000,1);
psd_est_aem = log(mean((est_s_test_aem),2));
% [p_aem,f] = pwelch(x_test_aem,hamming(500),0.6,n_fft,Fs_new);
% [p_iem,f] = pwelch(x_test_iem,hamming(500),0.6,n_fft,Fs_new);

% psd_est_aem = log(mean((s_aem_est.'),2));
p_aem = log(mean(abs(s_test_aem),2));
p_iem = log(mean(abs(s_test_iem),2));

MSSE_pred = sqrt(mean((p_aem-psd_est_aem).^2, 1));
MSSE_orig = sqrt(mean((p_aem-p_iem).^2, 1));



figure(3)
hold on
box on
plot(F, psd_est_aem, 'linewidth',2)
plot(F, log(mean(abs(s_test_aem),2)), 'linewidth',2)
plot(F, log(mean(abs(s_test_iem),2)), 'linewidth',2)
% plot(F, log(p_aem)-log(p_iem), 'linewidth',2)
xlabel('Frequency')
ylabel('Log-Power Spectra')
legend('Estimated AEM', 'AEM','IEM')
title('Log-Power Spectra of estimated AEM speech')
xlim([300, 3400])