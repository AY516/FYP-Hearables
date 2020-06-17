function [psd_est_aem, F, time, h_smooth] = batchTF(test_aem,test_iem)
addpath('/Users/anirudh/Desktop/FYP/sap-voicebox/voicebox');
%% Read files from text file 

training_size = 70;

aem_batchFile = 'train_aem.txt';
iem_batchFile = 'train_iem.txt';

aem_fileID = fopen(aem_batchFile,'r');
aem_cell = textscan(aem_fileID,'%s','Delimiter','\n');

iem_fileID = fopen(iem_batchFile,'r');
iem_cell = textscan(iem_fileID,'%s','Delimiter','\n');

aem_cell = aem_cell{1,1};
iem_cell = iem_cell{1,1};

%% STFT parameters
n_fft = 2048;
win_length = 400;
hop_length = 0.25*win_length;
smoothing_win_length = 10;
sampFreq = 16000; 

%% Train Transfer function 

for i=1:training_size
    
    aem_filepath = aem_cell{i,1};
    iem_filepath = iem_cell{i,1};
    
    % load AEM and IEM audio files
    [x_aem, fs] = audioread(aem_filepath); 
    [x_iem, fs] = audioread(iem_filepath); 
    
    %normalises
    Fs_new = 16000;
    [x_aem, x_iem, ~] = preproc(x_aem, x_iem, fs, Fs_new);
   
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
%% Evaluate estimated transfer function
[x_test_aem, fs] = audioread(test_aem); 
[x_test_iem, fs] = audioread(test_iem); 

[x_test_aem, x_test_iem, F] = preproc(x_test_aem, x_test_iem, fs, sampFreq);
[seg_test_aem, time] = v_enframe(x_test_aem,hamming(win_length,'periodic'),hop_length);
seg_test_iem = v_enframe(x_test_iem,hamming(win_length,'periodic'),hop_length);
time = time ./ F;

[nfrm_test,~] = size(seg_test_iem);
for i=1:nfrm_test
    [s_aem_frm, f, T, ~] = spectrogram(seg_test_iem(i,:), win_length, hop_length, n_fft, F, 'yaxis');
    s_aem_est(i,:) = h_smooth .* abs(s_aem_frm);
end
 
[s_test_aem, ~, ~, P_test_aem] = spectrogram(x_test_aem, win_length, hop_length, n_fft, Fs_new, 'yaxis');
[s_test_iem, F, T, P_test_iem] = spectrogram(x_test_iem, win_length, hop_length, n_fft, Fs_new, 'yaxis');
% 
% % apply learnt transfer function to the test set
% est_s_test_aem = h_smooth .* abs(s_test_iem);
est_P_aem = pow2db(s_aem_est);

psd_est_aem = psd(s_aem_est,win_length, sampFreq);
end

