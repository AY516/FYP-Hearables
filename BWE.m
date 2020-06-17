function [x_bwe,fs, est_x_aem] = BWE(x,fs, frm_length, lpc_order, cut_off)
% Bandwidth extension function
% Input: x_iem: In-ear Speech, fs: Sampling Frequency
%     : lpc_order: LPC filter Order, cut_off: filter cut-off frequency
% Output: x_bwe: Bandwidth Extended Signal, fs: Sampling frequency 


frm_idx = 1:frm_length:length(x);
x_bwe = [];

for i=1:length(frm_idx)-1
    
    up_x_iem = upsample(x(frm_idx(i):frm_idx(i)+frm_length-1), 2);
    lpc_coef = lpc(up_x_iem,lpc_order);

    est_x_aem = filter([0 -lpc_coef(2:end)],1,up_x_iem);
    
    % x^3 excitation signal
    ex = est_x_aem.^3;

    % band-extended excitation signal
    sum = ex + est_x_aem;
    
    % filtering stage
    [b_hpf, a_hpf] = butter(3,cut_off/8000,'high');
    [b_lpf, a_lpf] = butter(3, cut_off/8000);
    [b_bpf, a_bpf] = butter(4, [300/8000, 0.5]);
    
    hpf = filter(b_hpf, a_hpf, sum);
    lpf = filter(b_lpf, a_lpf, hpf);
    
    sum2 = hpf+lpf;

    sum2 = downsample(sum2,2);
    x_bwe = cat(1,x_bwe,sum2);
end
end
