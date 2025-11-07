import mne 
import matplotlib.pyplot as plt 

raw = mne.io.read_raw_edf("S001R01.edf", preload=True) 
print(raw.info) 

raw.filter(l_freq=1.0, h_freq=40.0)  # 1–40 Hz bandpass
raw.notch_filter(freqs=[60])   

raw.plot(n_channels=10, scalings='auto') 
plt.show()

# Plot PSD for all channels
raw.plot_psd(fmin=1, fmax=50, average=True)
plt.show()

# Or log-scaled with more detail
raw.plot_psd(fmin=1, fmax=50, average=True, dB=True)
plt.show()

raw = mne.io.read_raw_edf("S001R03.edf", preload=True) 
print(raw.info) 

raw.filter(l_freq=1.0, h_freq=40.0)  # 1–40 Hz bandpass
raw.notch_filter(freqs=[60])   

raw.plot(n_channels=10, scalings='auto') 
plt.show()

# Plot PSD for all channels
raw.plot_psd(fmin=1, fmax=50, average=True)
plt.show()

# Or log-scaled with more detail
raw.plot_psd(fmin=1, fmax=50, average=True, dB=True)
plt.show()
