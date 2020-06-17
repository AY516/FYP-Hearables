clear all; clc;
addpath('/Users/anirudh/Desktop/FYP/sap-voicebox/voicebox');
%% Parameters

rng(1); % For reproducibility

training_size = 70;
n_clusters = 10;
sampFreq = 16000;

n_fft = 2048;
win_length = 400;
hop_length = round(0.25*win_length);
smoothing_win_length = 10;

%lpc
lpc_order = 18;
n_lpcc = 18;

% training audio files filepath  
aem_batchFile = 'train_aem.txt';
iem_batchFile = 'train_iem.txt';

% test utterance file filepath
test_aem = '/Users/anirudh/Desktop/FYP/audio_files/aem/sp1_49_aem.flac';
test_iem = '/Users/anirudh/Desktop/FYP/audio_files/iem/sp1_49_iem.flac';

% test audio files filepath
% only used when performing batch testing 
test_aem_batchFile = 'test_aem.txt';
test_iem_batchFile = 'test_iem.txt';
%% Read files from text file 

aem_fileID = fopen(aem_batchFile,'r');
aem_cell = textscan(aem_fileID,'%s','Delimiter','\n');

iem_fileID = fopen(iem_batchFile,'r');
iem_cell = textscan(iem_fileID,'%s','Delimiter','\n');

aem_cell = aem_cell{1,1};
iem_cell = iem_cell{1,1};

%% Training 

train_aem = zeros(1,win_length);
train_iem = zeros(1,win_length);
for i=1:training_size
    
    aem_filepath = aem_cell{i,1};
    iem_filepath = iem_cell{i,1};
    
    % load AEM and IEM audio files
    [x_aem, fs] = audioread(aem_filepath); 
    [x_iem, fs] = audioread(iem_filepath); 
    
    %preprocessing: sampling, mmse_enh, Silence Removal, normalisation
    [x_aem, x_iem, F] = preproc(x_aem, x_iem, fs, sampFreq);
    
    %segment speech into overlapping frames
    seg_aem =v_enframe(x_aem,hamming(win_length,'periodic'),hop_length);
    seg_iem =v_enframe(x_iem,hamming(win_length,'periodic'),hop_length);
    
    %concatnate speech frames to make training vectors
    train_aem  = cat(1,train_aem,seg_aem);
    train_iem  = cat(1,train_iem,seg_iem);
end

train_aem = train_aem(2:end,:);
train_iem = train_iem(2:end,:);

dim_enframe = size(train_iem);
nfrm = dim_enframe(1);

for i=1:nfrm
    lpc_iem(i,:)  = lpc(train_iem(i,:),lpc_order);
    lpcc_iem(i,:)=v_lpcar2cc(lpc_iem(i,:),n_lpcc);
end
[idx,C] = kmeans(lpcc_iem, n_clusters, 'MaxIter',1000);

%% Compute average Cluster centre Transfer Functions

% Matrix containing average filter weights
% Dimensions : Number of CLusters x (NFFT/2 +1)
avg_cluster_weights = zeros(n_clusters, n_fft/2 +1);

% Iterate of each cluster
for i=1:n_clusters
    
    %find indicies of data frame in particular cluster
    cluster_idx = find(idx==i);
    
    iem_cluster = zeros(length(cluster_idx), win_length);
    aem_cluster = zeros(length(cluster_idx), win_length);
    cluster_w_mat = zeros(length(cluster_idx), n_fft/2 + 1);
    
    for j=1:length(cluster_idx)
        
        % get aem-iem pair from a particular cluster
        iem_cluster(j,:) = train_iem(cluster_idx(j),:);
        aem_cluster(j,:) = train_aem(cluster_idx(j),:);
        
        % Extract spectral features
        [s_aem, ~, ~, ~] = spectrogram(aem_cluster(j,:), win_length, hop_length, n_fft, F, 'yaxis');
        [s_iem, ~, T, ~] = spectrogram(iem_cluster(j,:), win_length, hop_length, n_fft, F, 'yaxis');
       
        %per-cluster per aem-iem pair tansfer function
        h = abs(s_aem)./abs(s_iem);
        h = smoothdata(h, 'movmean', smoothing_win_length);
        %matrix containing transfer function for all aem-iem instances in a
        %cluster
        cluster_h_mat(j,:) = h;
        
    end
    % matrix containing average cluster transfer function per cluster
    avg_cluster_weights(i,:) = mean(cluster_h_mat,1);
end


%% Testing 
% 
[x_test_aem, fs] = audioread(test_aem); 
[x_test_iem, fs] = audioread(test_iem);

[x_test_aem, x_test_iem, F] = preproc(x_test_aem, x_test_iem, fs, sampFreq);

[seg_test_aem, time] = v_enframe(x_test_aem,hamming(win_length,'periodic'),hop_length);
seg_test_iem =v_enframe(x_test_iem,hamming(win_length,'periodic'),hop_length);
time = time ./ F;

dim_test = size(seg_test_aem);
nfrm_test = dim_test(1);


% Test utterance classification loop

for i=1:nfrm_test
    %get LPCC coefficients of test data frames
    lpc_iem_test  = lpc(seg_test_iem(i,:),lpc_order); 
    lpcc_iem_test = v_lpcar2cc(lpc_iem_test, n_lpcc);
    
    %fit new data frames into the existing clusters
    %Euclidean dissimilarity to find closest cluster centre
    [~,idx_test] = pdist2(C,lpcc_iem_test,'euclidean','Smallest',1);
    
    %get corresponding average cluster weights for data frames
    h = avg_cluster_weights(idx_test,:);
    % apply smoothing window of length : smoothing_win_length
    h = smoothdata(h, 'movmean', smoothing_win_length);
    
    % get estimated of AEM speech frame using selected mapping h 
    [s_aem_frm, f, T, ~] = spectrogram(seg_test_iem(i,:), win_length, hop_length, n_fft, F, 'yaxis');
    
    % apply the classified transfer function to the test speech frame
    s_aem_est(i,:) = h .* abs(s_aem_frm).';
    si(i,:) = abs(s_aem_frm).';
    
    [s_aem, f, T, ~] = spectrogram(seg_test_aem(i,:), win_length, hop_length, n_fft, F, 'yaxis');
    sa(i,:) = abs(s_aem).';
end

[s_test_aem, f, T, ~] = spectrogram(x_test_aem, win_length, hop_length, n_fft, F, 'yaxis');
[s_test_iem, f, T, ~] = spectrogram(x_test_iem, win_length, hop_length, n_fft, F, 'yaxis');

% Obtain PSD from STFT-CMPLX coefficients 
psd_aem = psd(sa,win_length, sampFreq);
psd_iem = psd(si,win_length, sampFreq);
psd_est_aem = psd(s_aem_est,win_length, sampFreq);

%% Batch testing
% uncommet this section and comment section above for batch testing

% test_size = 11;
% 
% iem_test_fileID = fopen(test_aem_batchFile,'r');
% iem_test_cell = textscan(iem_test_fileID,'%s','Delimiter','\n');
% 
% aem_test_fileID = fopen(test_iem_batchFile,'r');
% aem_test_cell = textscan(aem_test_fileID,'%s','Delimiter','\n');
% 
% aem_test_cell = aem_test_cell{1,1};
% iem_test_cell = iem_test_cell{1,1};
% 
% msse = zeros(test_size, 3);
% 
% for j=1:test_size
% 
%     aem_test_filepath = aem_test_cell{j,1};
%     iem_test_filepath = iem_test_cell{j,1};
%     
% %     load AEM and IEM audio files
%     [x_test_aem, fs] = audioread(aem_test_filepath); 
%     [x_test_iem, fs] = audioread(iem_test_filepath); 
%     
% %     preprocessing: sampling, mmse_enh, Silence Removal, normalisation
%     [x_test_aem, x_test_iem, F] = preproc(x_test_aem, x_test_iem, fs, sampFreq);
%     
% %     segment speech into overlapping frames
%     seg_test_aem =v_enframe(x_test_aem, hamming(win_length,'periodic'), hop_length);
%     seg_test_iem =v_enframe(x_test_iem, hamming(win_length,'periodic'), hop_length);
%     
%     [nfrm_test, ~] = size(seg_test_aem);
%     
%     for i=1:nfrm_test
%         
% %         get lpc coefficients of test data frames
%         lpc_iem_test  = lpc(seg_test_iem(i,:),lpc_order); 
%         lpcc_iem_test = v_lpcar2cc(lpc_iem_test, n_lpcc);
% 
% %         fit new data frames into the existing clusters
%         [~,idx_test] = pdist2(C,lpcc_iem_test,'euclidean','Smallest',1);
% 
% %         get corresponding average cluster weights for data frames
%         h = avg_cluster_weights(idx_test,:);
%         h = smoothdata(h, 'movmean', smoothing_win_length);
% 
% %         get estimated of AEM speech frame using selected mapping h 
%         [s_aem_frm, f, T, ~] = spectrogram(seg_test_iem(i,:), win_length, hop_length, n_fft, F, 'yaxis');
%         s_aem_est(i,:) = h .* abs(s_aem_frm).';
% 
%         si(i,:) = abs(s_aem_frm).';
%         [s_aem, f, T, ~] = spectrogram(seg_test_aem(i,:), win_length, hop_length, n_fft, F, 'yaxis');
%         sa(i,:) = abs(s_aem).';
%     end
%     
%     [s_test_aem, f, T, ~] = spectrogram(x_test_aem, win_length, hop_length, n_fft, F, 'yaxis');
%     [s_test_iem, f, T, ~] = spectrogram(x_test_iem, win_length, hop_length, n_fft, F, 'yaxis');
%         
% %     PSD of Phoneme-TF
%     psd_aem = psd(sa,win_length, sampFreq);
%     psd_iem = psd(si,win_length, sampFreq);
%     psd_est_aem = psd(s_aem_est,win_length, sampFreq);
%     
% %     PSD Spectral-Gain TF
%     [s, F, T] = batchTF(aem_test_filepath, iem_test_filepath);
%     
% %     Mean-Square Spectral Error
%     Range = [0, 2000; 2000,4000; 4000, 8000; 0,8000];
%     [ndiv,~] = size(Range);
%     for div= 1:ndiv
%         FreqRange = Range(div,:);
%         RangeBin=freq2bin(FreqRange,sampFreq/2,n_fft/2);
%         RangeBin=RangeBin(1):RangeBin(2);
%         psd_est_aem_clus = log(mean(abs((psd_est_aem(1:nfrm_test, RangeBin).')),2));
%         psdPlot_est_aem = log(mean(abs(s(1:nfrm_test, RangeBin).'),2));
%         paem = log(mean(abs(psd_aem(1:nfrm_test, RangeBin).'),2));
%         piem = log(mean(abs(psd_iem(1:nfrm_test, RangeBin).'),2));
% 
%         MSSE_psd_orig = sqrt(mean((paem - piem).^2,1));
%         MSSE_psd_ltf = sqrt(mean((paem - psdPlot_est_aem).^2,1));
%         MSSE_psd_clus = sqrt(mean((paem - psd_est_aem_clus).^2,1));
% 
%         [~,filename,~] = fileparts(iem_test_filepath);
%         msse_psd_orig(div,j) = MSSE_psd_orig;
%         msse_psd_ltf(div,j) = MSSE_psd_ltf;
%         msse_psd_cls(div,j) = MSSE_psd_clus;
%     end
%     file(:,j) = filename;
% end
%% Baseline Method : Avg-Gain Eq Approach
[s, F, T, h_smooth] = batchTF(test_aem, test_iem);
%% Plots and Figures 
figure(2)
sgtitle ('\fontsize{16}IEM vs Spectral Methods Comparision')
subplot(4,1,1)
hold on
box on
imagesc(time,f, pow2db(psd_est_aem.'))
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Phoneme-based Gain Estimated AEM')

subplot(4,1,2)
hold on
box on
imagesc(T,f, pow2db(s.'))
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Average-Spectral Gain Estimated AEM')

subplot(4,1,3)
hold on
box on
imagesc(time,f, pow2db(psd_aem.'))
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('External Microphone(AEM)')

subplot(4,1,4)
hold on
box on
imagesc(time,f, pow2db(psd_iem.'))
axis tight;
view(2);
colormap('jet')
c = colorbar;
c.Label.String = "dB/Hz";
ylim([300 3400])
ylabel('Frequency (Hz)')
xlabel('Time')
title('Internal Microphone(IEM)')

% log-PSD

p_est_aem = log(mean(abs((s_aem_est.')),2));
p_aem = log(mean(abs(s_test_aem),2));
% p_aem = log(mean(abs(sa.'),2));
p_iem = log(mean(abs(s_test_iem),2));
% p_iem = log(mean(abs(si.'),2));
%Mean-Square Spectral Error
MSSE_pred = sqrt(mean((p_aem-p_est_aem).^2, 1));
MSSE_orig = sqrt(mean((p_aem-p_iem).^2, 1));


% Plot Cluster Transfer functions
% figure(4)
% subplot(1,2,2)
% plot(f,pow2db(avg_cluster_weights.'))
% xlabel('Frequency')
% ylabel('Amplitude (dB)')
% title('Cluster Centres: EQ Transfer function')
% 
% subplot(1,2,1)
% plot(f,pow2db(h_smooth))
% xlabel('Frequency')
% ylabel('Amplitude (dB)')
% title('Linear Gain Average Transfer function')
% xlim([0, 3400])
%%
psd_est_aem_clus = log(mean(abs((psd_est_aem.')),2));
psdPlot_est_aem = log(mean(abs(s.'),2));
paem = log(mean(abs(psd_aem.'),2));
piem = log(mean(abs(psd_iem.'),2));

MSSE_psd_orig = sqrt(mean((paem - piem).^2,1));
MSSE_psd_ltf = sqrt(mean((paem - psdPlot_est_aem).^2,1));
MSSE_psd_clus = sqrt(mean((paem - psd_est_aem_clus).^2,1));

figure(5)
sgtitle ('\fontsize{14} Log-PSD comparision of Phoneme vs Linear Gain')
subplot(1,2,1)
box on
hold on
plot(f, psd_est_aem_clus, 'linewidth',2);
plot(f, log(mean(abs(psd_aem.'),2)), 'linewidth',2);
plot(f, log(mean(abs(psd_iem.'),2)), 'linewidth',2);
xlim([300,8000])
xlabel('\fontsize{14} Frequency (Hz)')
ylabel('\fontsize{14} Log-Power Spectral Density (dB)')
legend('\fontsize{12} Estimated AEM - Phoneme Map','\fontsize{12}AEM','\fontsize{12}IEM')
title(['Log-PSD of Estimated AEM - Phoneme Map', ', ', 'LSD: ', num2str(MSSE_psd_clus)])
set(gca, 'Fontsize',12)

subplot(1,2,2)
box on
hold on
plot(f, psdPlot_est_aem, 'linewidth',2);
plot(f, log(mean(abs(psd_aem.'),2)), 'linewidth',2);
plot(f, log(mean(abs(psd_iem.'),2)), 'linewidth',2);
% plot(f, log(mean(abs(psd_aem.'),2)) - psdPlot_est_aem, 'linewidth', 2);
xlim([300,8000])
xlabel('\fontsize{14} Frequency (Hz)')
ylabel('\fontsize{14} Log-Power Spectral Density (dB)')
legend('\fontsize{12} Estimated AEM - Linear Gain','\fontsize{12} AEM','\fontsize{12}IEM')
title(['Log-PSD of Estimated AEM - Linear Gain', ', ', 'LSD: ', num2str(MSSE_psd_ltf)])
set(gca, 'Fontsize',12)

hold off
% plot(pow2db(cluster_h_mat(1,:).'))


%% LSD spectral mismatch
% sgtitle ('\fontsize{14} Log-PSD comparision of Phoneme vs Linear Gain')
% subplot(2,2,1)
% box on
% hold on
% set(gca, 'Fontsize',12)
% plot(f, psd_est_aem_clus, 'linewidth',2);
% plot(f, log(mean(abs(psd_aem.'),2)), 'linewidth',2);
% plot(f, log(mean(abs(psd_iem.'),2)), 'linewidth',2);
% xlim([300,8000])
% xlabel('\fontsize{14} Frequency (Hz)')
% ylabel('\fontsize{14} Log-PSD (dB)')
% legend('\fontsize{12} P-M','\fontsize{12}AEM','\fontsize{12}IEM')
% title('\fontsize{14}Log-PSD: AEM - P-M')
% 
% 
% subplot(2,2,2)
% box on
% hold on
% set(gca, 'Fontsize',12)
% plot(f, psdPlot_est_aem, 'linewidth',2);
% plot(f, log(mean(abs(psd_aem.'),2)), 'linewidth',2);
% plot(f, log(mean(abs(psd_iem.'),2)), 'linewidth',2);
% xlim([300,8000])
% xlabel('\fontsize{14} Frequency (Hz)')
% ylabel('\fontsize{14} Log-PSD (dB)')
% legend('\fontsize{12} SG-AEM','\fontsize{12} AEM','\fontsize{12}IEM')
% title('\fontsize{14} Log-PSD: SG-AEM')
% 
% 
% subplot(2,2,3)
% box on
% hold on
% set(gca, 'Fontsize',12)
% plot(f, log(mean(abs(psd_aem.'),2))- psd_est_aem_clus, 'linewidth',2);
% plot(f, log(mean(abs(psd_aem.'),2))- log(mean(abs(psd_iem.'),2)), 'linewidth',2);
% xlim([300,8000])
% xlabel('\fontsize{14} Frequency (Hz)')
% ylabel('\fontsize{14} Log-PSD (dB)')
% legend('\fontsize{12}AEM- P-M','\fontsize{12} AEM vs. IEM','\fontsize{12}IEM')
% title('\fontsize{14} Spectral Mismatch: P-M')
% 
% subplot(2,2,4)
% box on
% hold on
% set(gca, 'Fontsize',12)
% plot(f, log(mean(abs(psd_aem.'),2)) - psdPlot_est_aem, 'linewidth', 2);
% plot(f, log(mean(abs(psd_aem.'),2)) - log(mean(abs(psd_iem.'),2)), 'linewidth', 2);
% xlim([300,8000])
% xlabel('\fontsize{14} Frequency (Hz)')
% ylabel('\fontsize{14} Log-PSD (dB)')
% legend('\fontsize{12} SG-AEM','\fontsize{12} AEM vs. IEM','\fontsize{12}IEM')
% title('\fontsize{14} Spectral Mismatch: SG-AEM')
% 
% hold off
