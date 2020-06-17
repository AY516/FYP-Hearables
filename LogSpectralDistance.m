function LSD=LogSpectralDistance(Clean,Noisy,fs,RS,Range)
%LSD=LOGSPECTRALDISTANCE(CLEAN,NOISY,FS,RS,RANGE)
% Calculates the average log-spectral distance between CLEAN and NOISY
% signals. Frames are 25ms long with 60 percent (15ms) overlap, hamming
% windowed. RS is the remove silence option (default: 0) if 1, the program
% uses a Voice Activity Detector to find non-speech frames and eliminates
% them from calculation of LSD. FS is the sampling frequency (Default
% 10KHz). RANGE is the frequency range used in the calculation of LSD
% (default: [0 FS/2]) it is a two-element vector the first element should
% be smaller than the second one and non of them should be greater than
% FS/2.
if nargin<3
    fs=10000;
end
if nargin<4
    RS=0;
end
if nargin<5
    Range=[0,fs/2];
end
if RS==1
    [Clean, Noisy]=RemoveSilence(Clean,Noisy,fs);
end
Len=min(length(Clean),length(Noisy));
Clean=Clean(1:Len);
Noisy=Noisy(1:Len);
Clean=Clean./sqrt(sum(Clean.^2));
Noisy=Noisy./sqrt(sum(Noisy.^2));
W=round(.025*fs);
SP=.4;
CL=abs(mySpecgram(Clean,W,SP));
NO=abs(mySpecgram(Noisy,W,SP));
nfft2=size(CL,1);
N=min(size(CL,2),size(NO,2)); %Number of frames
RangeBin=freq2bin(Range,fs/2,nfft2);
RangeBin=RangeBin(1):RangeBin(2);

pc = log(CL(RangeBin,1:N));
pn = log(NO(RangeBin,1:N));
avg = mean(pc - pn);

LSD=mean(sqrt(mean((log(CL(RangeBin,1:N))-log(NO(RangeBin,1:N))).^2)));


