import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt  # or "Qt5Agg"
import mne

troubleshoot_epochs = False

mne.viz.set_browser_backend("qt")
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

# 4) Bad channels → interpolate, then re-reference (CAR)
# (quick heuristic, or mark manually if you know them)
raw.info["bads"] = []  # set if you know bads; else skip
raw.interpolate_bads(reset_bads=True)
raw.set_eeg_reference("average")

# Map T0 -> 0 (rest), T1/T2 -> 1 (hand/movement)
mapping = {}
for d in set(raw.annotations.description):
    if "T0" in d: mapping[d] = 0
    if "T1" in d or "T2" in d: mapping[d] = 1

events, _ = mne.events_from_annotations(raw, event_id=mapping)
epochs = mne.Epochs(raw, events, event_id=mapping,
                    tmin=-0.5, tmax=3.0, baseline=(-0.5, 0.0),
                    picks="eeg", preload=True)

# 6) Arrays for ML
X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs.events[:, -1].astype(int)  # 0=rest, 1=hand

print(X.shape, y.shape)




####TROUBLESHOOTING#######
if troubleshoot_epochs:
    print("Annotations:", set(raw.annotations.description))
    print("sfreq:", raw.info["sfreq"], "duration (s):", raw.times[-1])

    events, _ = mne.events_from_annotations(raw, event_id=mapping)
    print("Events shape:", events.shape)
    print("Event times (s):", events[:, 0] / raw.info["sfreq"])

    # After constructing epochs:
    print("tmin,tmax:", epochs.tmin, epochs.tmax)
    print("reject_by_annotation:", epochs.reject_by_annotation)
    print("Drop log (first 10):", epochs.drop_log[:10])   # reason per event
    # In newer MNE: use epochs.drop_log; there is no attribute 'drop_indices'.
    # Kept indices are in:
    print("Kept indices (epochs.selection):", getattr(epochs, "selection", None))
    # Visual:
    # epochs.plot_drop_log()