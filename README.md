# EEG Motor Imagery Classification

This project explores **EEG (Electroencephalography)** data to visualize and analyze brain activity during **motor imagery tasks** — mental simulation of movement (e.g., imagining moving your fists vs moving your feet). The goal is to understand how brainwave patterns change across different mental states, and predict imagined movement using **AI classification**.

---

## Required Libraries
os,
keras,
numpy,
mne,
matplotlib,
sklearn

## How to Run
To run the within subject, simply run withinSubject.py in your IDE. The 
program should automatically download the subject from the PhysioNet dataset, 
and run through the training

For cross subject, a model has been pre-trained for the sake of time.
To run on this pre-trained model, simply run "runCrossSubModel.py", and the 
program will download the necessary data from the PhysioNet database.

To train a new cross subject model, run "crossSubject.py". Note that this may take a while,
as it needs to both download the dataset (which is around 3GB) and then train.


## Project Overview

* **Dataset:** [PhysioNet EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
* **Format:** EDF (European Data Format)
* **Tools Used:**

  * [MNE-Python](https://mne.tools/stable/index.html) for EEG signal processing
  * Matplotlib for visualization
  * NumPy for data handling
---

## 📜 License

This project is open for educational and research purposes.
EEG data courtesy of the PhysioNet EEG Motor Movement/Imagery dataset.
