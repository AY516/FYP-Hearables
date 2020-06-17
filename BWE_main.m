clear all; clc;
%% 
n_fft = 2048;
win_length = 400;
hop_length = round(0.25*win_length);

% input aem-iem filepath
filepath_aem = '/Users/anirudh/Desktop/FYP/audio_files/aem/sp1_44_aem.flac';
filepath_iem = '/Users/anirudh/Desktop/FYP/audio_files/iem/sp1_44_iem.flac';

% save BWE audio file path
filepath_bwe = '/Users/anirudh/Desktop/FYP/code/aem_est.wav';

%% Initialisation
Fs_new = 16000;
% 
[x_aem, fs] = audioread(filepath_aem); 
[x_iem, fs] = audioread(filepath_iem);

[Numer, Denom] = rat(Fs_new/fs);
x_aem = resample(x_aem, Numer, Denom);
x_iem = resample(x_iem, Numer, Denom); 

[x_aem,x_iem] = RemoveSilence(x_aem,x_iem,Fs_new);
x_aem = v_ssubmmse(x_aem, Fs_new);
x_iem = v_ssubmmse(x_iem, Fs_new);
% 
x_aem = x_aem - mean(x_aem);
x_iem = x_iem - mean(x_iem);
% % 
x_aem = x_aem./max(x_aem);
x_iem = x_iem./max(x_iem);

%% BWE
frm_length = 4000;
[aem_est, ~, est_x_aem] = BWE(x_iem, Fs_new, frm_length, 18, 1800);
aem_est = v_ssubmmse(aem_est, Fs_new);

aem_est = aem_est - mean(aem_est);
aem_est = aem_est./max(aem_est);
audiowrite(filepath_bwe,aem_est, Fs_new)
sound(aem_est,Fs_new)

min_length = length(aem_est);

[s_est_aem, F, T, P_est_aem] = spectrogram(aem_est, win_length, hop_length, n_fft, Fs_new, 'yaxis');
[s_aem, F, T, P_aem] = spectrogram(x_aem(1:min_length), win_length, hop_length, n_fft, Fs_new, 'yaxis');
[s_iem, F, T, P_iem] = spectrogram(x_iem(1:min_length), win_length, hop_length, n_fft, Fs_new, 'yaxis');


[est, fs_est] = audioread(filepath_bwe); 
LSD_orig = LogSpectralDistance(x_aem,x_iem, 16000,1);
LSD_tf = LogSpectralDistance(x_aem,est,16000,1);

%% Plots
figure(2)
box on
sgtitle('Artificial Bandwidth Extension of IEM speech')
subplot(3,1,1)
surf(T,F,pow2db(P_est_aem), "LineStyle", "none")
view(2) 
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
xlim([T(1) T(end)])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Bandwidth Extended IEM Speech')

subplot(3,1,2)
surf(T,F,pow2db(P_aem), "LineStyle", "none")
view(2) 
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
xlim([T(1) T(end)])
ylabel('Frequency (Hz)')
xlabel('Time')
title('External Microphone (AEM)')

subplot(3,1,3)
surf(T,F,pow2db(P_iem), "LineStyle", "none")
view(2) 
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
xlim([T(1) T(end)])
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('In-ear Microphone (IEM)')
