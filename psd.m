function P = psd(s,win, fs)
%Return PSD of given the CMPLX Fourier coefficients
% and the Window length. Default window: Hamming

p_signal = abs(s.^2);
p_win = hamming(win).'*hamming(win);
P = p_signal ./ (fs * p_win);
end

