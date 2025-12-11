#####################################################################################
# Clean within-subject EEGNet pipeline with sliding windows and robust event handling.
# By: Michael Beehler, Javon Bell, Endi Troqe
# Last Edited: 12/11/2025 (By Michael Beehler)
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet 

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------
# USER CONFIG
# -------------------------
MNE_DATA_PATH = r'c:\Users\Michael\Desktop\EEGData'
SUBJECT = 10                      # single subject for within-subject model
RUNS = [6, 10, 14]               # motor imagery hands vs feet (adjust if you want other tasks)
TMIN, TMAX = 0.0, 4.0           # epoch window (kept large; training windows will be cropped)
TRAIN_WINDOW_SIZE = 160          # samples per sliding window (EEGNet Samples parameter)
STEP_SIZE = 40                   # sliding window step (overlap = window_size - step_size)
RESAMPLE_SFREQ = 128             # EEGNet expects 128 Hz
BATCH_SIZE = 16
EPOCHS = 100
CHECKPOINT_PATH = "/tmp/checkpoint.h5"
SEED = 42
np.random.seed(SEED)

# -------------------------
# Setup MNE data path
# -------------------------
mne.set_config('MNE_DATA', MNE_DATA_PATH)

# -------------------------
# Load raw EDFs for subject
# -------------------------
raw_fnames = eegbci.load_data(SUBJECT, RUNS)
raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
raw = concatenate_raws(raws)

# standardize channel names & montage
eegbci.standardize(raw)
montage = make_standard_montage("standard_1005")
raw.set_montage(montage)

# re-reference (projection)
raw.set_eeg_reference(projection=True)

# bandpass
raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

# resample to EEGNet expected sampling rate
raw.resample(RESAMPLE_SFREQ, npad="auto")

# picks
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

# -------------------------
# Robust event mapping
# -------------------------
# Use annotations to get the available event/descriptions and integer codes
events, annot_map = mne.events_from_annotations(raw)
print("Annotation -> integer mapping found in file:", annot_map)
# annot_map example: {'T0': 1, 'T1': 2, 'T2': 3} or {'hands': 2, 'feet':3} depending on renaming

# We want to pick the two descriptions relating to hands vs feet for runs [6,10,14]
# Heuristic: prefer explicit 'hand'/'feet' or 'T1'/'T2' labels. If not found, print and abort.
def find_label_keys(map_dict):
    keys = list(map_dict.keys())
    hands_key = None
    feet_key = None
    for k in keys:
        kl = k.lower()
        if 'hand' in kl or 'hands' in kl or 'left' in kl:
            hands_key = k
        if 'foot' in kl or 'feet' in kl or 'right' in kl:
            feet_key = k
    # fallback to T1/T2
    if hands_key is None and 't1' in [k.lower() for k in keys]:
        for k in keys:
            if k.lower() == 't1':
                hands_key = k
    if feet_key is None and 't2' in [k.lower() for k in keys]:
        for k in keys:
            if k.lower() == 't2':
                feet_key = k
    return hands_key, feet_key

hands_key, feet_key = find_label_keys(annot_map)
print("Detected hands_key =", hands_key, "feet_key =", feet_key)

if hands_key is None or feet_key is None:
    raise RuntimeError("Could not automatically find hands/feet annotation keys. Check annot_map printed above.")

event_id = { 'hands': annot_map[hands_key], 'feet': annot_map[feet_key] }
print("Using event_id mapping:", event_id)

# -------------------------
# Build epochs (full trial window)
# -------------------------
epochs = Epochs(raw, event_id=event_id, tmin=TMIN, tmax=TMAX,
                proj=True, picks=picks, baseline=None, preload=True)

print("Number of epochs:", len(epochs))
print("Event codes present (unique):", np.unique(epochs.events[:, -1]))

# Map numeric event codes to 0/1 labels (hands -> 0, feet -> 1)
code_to_label = { event_id['hands']: 0, event_id['feet']: 1 }
labels = np.array([code_to_label[e] for e in epochs.events[:, -1]])
print("Label distribution (counts):", np.bincount(labels))

# training epochs
X_full = epochs.get_data()   # shape: (n_trials, n_channels, n_samples)
print("Epoch data shape (trials, chans, samples):", X_full.shape)

# -------------------------
# Sliding-window extraction
# -------------------------
def sliding_window(data, labels, window_size=TRAIN_WINDOW_SIZE, step_size=STEP_SIZE):
    X_windows = []
    y_windows = []
    for ti in range(data.shape[0]):
        trial = data[ti]      # (chans, samples)
        label = labels[ti]
        n_samples = trial.shape[-1]
        start = 0
        while start + window_size <= n_samples:
            w = trial[:, start:start+window_size]
            X_windows.append(w)
            y_windows.append(label)
            start += step_size
    X_windows = np.array(X_windows)  # (n_windows, chans, window_size)
    y_windows = np.array(y_windows)
    # add channel-last singleton for EEGNet: (n_windows, chans, window_size, 1)
    X_windows = X_windows[..., np.newaxis]
    return X_windows, y_windows

X_win, y_win = sliding_window(X_full, labels, window_size=TRAIN_WINDOW_SIZE, step_size=STEP_SIZE)
print("Sliding windows shape (n_windows, chans, samples, 1):", X_win.shape)
print("Sliding windows label counts:", np.bincount(y_win))

if X_win.shape[0] < 10:
    raise RuntimeError("Too few windows produced; adjust window_size/step_size or epoch crop.")

# -------------------------
# Train / Val / Test splits (stratified)
# -------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_win, y_win, test_size=0.20, random_state=SEED, stratify=y_win)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=SEED, stratify=y_trainval)

print("Shapes: X_train, X_val, X_test ->", X_train.shape, X_val.shape, X_test.shape)
print("Class balance: train", np.bincount(y_train), "val", np.bincount(y_val), "test", np.bincount(y_test))

# -------------------------
# Prepare labels for EEGNet
# -------------------------
# labels are 0..nb_classes-1
nb_classes = 2
assert set(np.unique(y_train)).issubset({0,1})

y_train_hot = np_utils.to_categorical(y_train, nb_classes)
y_val_hot   = np_utils.to_categorical(y_val, nb_classes)
y_test_hot  = np_utils.to_categorical(y_test, nb_classes)

# EEGNet expects shape (n_epochs, chans, samples, 1)
chans = X_train.shape[1]
samples = X_train.shape[2]
print("EEGNet input dims -> chans:", chans, "samples:", samples)

# -------------------------
# Build and compile EEGNet
# -------------------------
model = EEGNet(nb_classes=nb_classes,
               Chans=chans,
               Samples=samples,
               dropoutRate=0.5,
               kernLength=32,
               F1=8, D=2, F2=16,
               dropoutType='Dropout')

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print("Model params:", model.count_params())

# callbacks
checkpointer = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_best_only=True)

# -------------------------
# Fit
# -------------------------
history = model.fit(X_train, y_train_hot,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=(X_val, y_val_hot),
                    callbacks=[checkpointer])

# -------------------------
# Evaluate on test set
# -------------------------
# Load the best weights, make predictions
model.load_weights(CHECKPOINT_PATH)
probs = model.predict(X_test)
preds = probs.argmax(axis=1)
y_true = y_test

test_acc = (preds == y_true).mean()
print(f"Test accuracy: {test_acc:.3f}")

# Plot and display a confusion matrix
cm = confusion_matrix(y_true, preds, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=['hands','feet'])
disp.plot()
plt.show()