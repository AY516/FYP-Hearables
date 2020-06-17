function X=mySpecgram(x,W,SP)
if nargin<3
    SP=.4;
end
if nargin<2
    W=250;
end
X=segment(x,W,SP);
X=fft(X);
X=X(1:fix(end/2)+1,:);
end