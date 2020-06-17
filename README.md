FYP-Hearables

Repositorty contains code for the Hearables Final Year Project. 
The project aims to develop an inverse transfer function to map the In-ear microphone speech to that from an external microphone. 


The key functions are as follows:
BWE.m : Artificial Bandwidth Extension Function.m
BWE_main.m : Executable code for BWE.m

phoneme_cluster.m : Phoneme-mapping Transfer Function
batchTF.m : Baseline Spectral Gain Transfer function. 

phoneme_cluster.m outputs comparision of both the baseline and the phoneme-mapping transfer function

Note: "v_" prefix code is adapted from VOICEBOX[1].

vad.m ,overlapadd2.m, segment.m, RemoveSilence.m and LogSpectralDistortion.m adapted from MathWork FileExchange.[2]

[1] Mike Brookes.  VOICEBOX: Speech Processing Toolbox for MATLAB, 2011.  retrieved on 8 June 2020 from:http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html.

[2] Esfandiar Zavarehei. Log spectral distance - silence removal, 2020. retrieved from MATLAB Central File  Exchange,https://www.mathworks.com/matlabcentral/fileexchange/9998-log-spectral-distance.
