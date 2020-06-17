function [x2, y2]=RemoveSilence(x1,y1,fs,IS)
% Remove Silence
% [X2, Y2]=REMOVESILENCE(X1,Y1,FS,IS)
% This function removes the silence parts from the signals X1 and Y1 and
% returns the corresponding wave forms. Y1 is normally a modified (noisy or
% enhanced) version of X1. The silence frames are detected using X1 which
% is supposed to be the clean signal. For this purpose a Voice Activity
% Detector (VAD) is used. The use of this function is for evaluation of the
% speech quality (e.g. SNR) at speech active frame only. FS is tha sampling
% frequency. IS is the initial silence duration (the defaul value) which is
% used to model the noise template. the default value of IS is 0.25 sec.
% Date: Feb-05
% Author: Esfandiar Zavarehei
if (nargin<4)
    IS=.25; %seconds
end
% Window size and overlap and other initialization values
W=.025*fs;
SP=.010*fs/W;
wnd=hamming(W);
NIS=fix((IS*fs-W)/(SP*W) +1);%number of initial silence segments
Y1=segment(y1,W,SP,wnd);
Y1=fft(Y1);
Y1=Y1(1:fix(end/2)+1,:);
Y1P=angle(Y1);
Y1=abs(Y1);
X1=segment(x1,W,SP,wnd);
X1=fft(X1);
X1=X1(1:fix(end/2)+1,:);
X1P=angle(X1);
X1=abs(X1);
NumOfFrames=min(size(X1,2),size(Y1,2));
NoiseLength=15;
N=mean(X1(:,1:NIS)')'; %initial Noise Power Spectrum mean
%Find the non-speech frames of X1
for i=1:NumOfFrames
    if i<=NIS
        SpeechFlag(i)=0;
        NoiseCounter=100;
        Dist=.1;
    else
        [NoiseFlag, SpeechFlag(i), NoiseCounter, Dist]=vad(X1(:,i),N,NoiseCounter,2.5,fix(.08*10000/(SP*W))); %Magnitude Spectrum Distance VAD
    end
    if SpeechFlag(i)==0 & i>NIS
        N=(NoiseLength*N+X1(:,i))/(NoiseLength+1); %Update and smooth noise mean
    end
end
SpeechIndx=find(SpeechFlag==1);
x2=OverlapAdd2(X1(:,SpeechIndx),X1P(:,SpeechIndx),W,SP*W);
y2=OverlapAdd2(Y1(:,SpeechIndx),Y1P(:,SpeechIndx),W,SP*W);
end
