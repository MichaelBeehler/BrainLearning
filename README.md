# 🧠 EEG Motor Imagery Classification

This project explores **EEG (Electroencephalography)** data to visualize and analyze brain activity during **motor imagery tasks** — mental simulation of movement (e.g., imagining moving your left or right hand). The goal is to understand how brainwave patterns change across different mental states, and predict real or imagined movement using **AI classification**.

---

## 🚀 Project Overview

* **Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
* **Format:** EDF (European Data Format)
* **Tools Used:**

  * [MNE-Python](https://mne.tools/stable/index.html) for EEG signal processing
  * Matplotlib for visualization
  * NumPy for data handling

This project currently focuses on:

1. Loading raw EEG data and inspecting metadata
2. Visualizing time-domain signals and spectral power
3. Cleaning and preprocessing EEG (filtering, referencing, artifact removal)
4. Comparing resting vs. motor imagery states

---

## 🧩 Example Workflow

```python
import mne
import matplotlib.pyplot as plt

# Load EEG data
raw = mne.io.read_raw_edf("S001R01.edf", preload=True)

# Apply filters
raw.filter(l_freq=1.0, h_freq=40.0)
raw.notch_filter(freqs=[60])

# Set montage (electrode positions)
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')

# Plot signals and PSD
raw.plot(n_channels=10, scalings='auto')
raw.plot_psd(fmin=1, fmax=50, average=True, dB=True)
plt.show()
```

---

## 📊 Planned Next Steps

* Implement event-based segmentation (epoching)
* Extract EEG features (alpha/beta power, CSP)
* Train a machine learning model to classify motor imagery (left vs. right hand)
* Generate topographic brain maps for visualization

---

## ⚙️ Requirements

```bash
pip install mne matplotlib numpy
```

---

## 📜 License

This project is open for educational and research purposes.
EEG data courtesy of the PhysioNet EEG Motor Movement/Imagery dataset.
