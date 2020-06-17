function [x_aem,x_iem, Fs_new] = preproc(x_aem,x_iem, fs, downsample_fs)
% Preprocessing module function
% Input: x_aem : REF speech, x_iem : In-ear Speech
%      : fs : original Samp. Frequency, downsample_fs: required downsample fs. 
% Output : x_aem, x_iem : downsampled, preprocessed speech
%        : Fs_new: downsample fs 

Fs_new = downsample_fs;

% resampling speech to downsample_fs
[Numer, Denom] = rat(Fs_new/fs);
x_aem = resample(x_aem, Numer, Denom);
x_iem = resample(x_iem, Numer, Denom);

% STSA-MMSE algorithm 
x_aem = v_ssubmmse(x_aem, Fs_new);
% x_iem = v_ssubmmse(x_iem, Fs_new);

% Voice Activity Dectection Silence Removal
[x_aem,x_iem] = RemoveSilence(x_aem,x_iem,Fs_new);
 
% Peak Normalisation
x_aem = x_aem - mean(x_aem);
x_iem = x_iem - mean(x_iem);

x_aem = x_aem./max(x_aem);
x_iem = x_iem./max(x_iem);

end

