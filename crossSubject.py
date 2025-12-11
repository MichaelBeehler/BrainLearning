"""
Cross-subject EEGNet training script (leave-one-subject-out).
- Trains on SUBJECTS excluding TEST_SUBJECT and evaluates on TEST_SUBJECT
- Robust event mapping per-subject
- Sliding windows to create many training samples
- Uses EEGNet from your EEGModels module (assumed to be Keras)
"""

import os
import numpy as np
import mne
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from EEGModels import EEGNet  # your local EEGNet implementation
import matplotlib.pyplot as plt

# -----------------------
# USER CONFIG
# -----------------------

# Base project folder (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where to store/download the EEG data
MNE_DATA_PATH = os.getenv("MNE_DATA_PATH",
                          os.path.join(BASE_DIR, "MNE_DATA"))

# Where to save model checkpoints
CHECKPOINT_PATH = os.path.join(BASE_DIR, "eegnet_best.h5")

# Apply path to MNE
mne.set_config("MNE_DATA", MNE_DATA_PATH)

SUBJECTS = list(range(1, 50))       # subjects to include (1..109)
TEST_SUBJECT = 4                   # leave this subject out for testing (cross-subject eval)
RUNS = [6, 10, 14]                   # motor imagery hands vs feet (adjust if needed)
TMIN, TMAX = 0.0, 4.0                # full epoch window (we'll extract sliding windows from this)
RESAMPLE_SFREQ = 128                 # EEGNet usually expects 128 Hz
WINDOW_SIZE = 160                    # samples in sliding window (EEGNet Samples param)
STEP_SIZE = 40                       # sliding window step (overlap = WINDOW_SIZE - STEP_SIZE)
BATCH_SIZE = 32
EPOCHS = 60
#CHECKPOINT_PATH = "/tmp/eegnet_best.h5"
SEED = 42
np.random.seed(SEED)

# -----------------------
# Helpers
# -----------------------
def find_hand_feet_keys(annot_map):
    """Return (hands_key, feet_key) from annotations mapping. Can be T1/T2 or 'hands','feet' etc."""
    keys = list(annot_map.keys())
    hands_key = None
    feet_key = None
    for k in keys:
        kl = k.lower()
        if any(w in kl for w in ("hand", "hands", "left")):
            hands_key = k
        if any(w in kl for w in ("foot", "feet", "right")):
            feet_key = k
    # fallback to T1/T2 if present
    if hands_key is None and any(k.lower() == "t1" for k in keys):
        hands_key = next(k for k in keys if k.lower() == "t1")
    if feet_key is None and any(k.lower() == "t2" for k in keys):
        feet_key = next(k for k in keys if k.lower() == "t2")
    return hands_key, feet_key

def sliding_windows_from_epochs(X_epochs, labels, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    """
    X_epochs: (n_trials, n_chans, n_samples)
    labels: (n_trials,) with values in {0,1}
    returns X_windows: (n_windows, n_chans, window_size, 1), y_windows: (n_windows,)
    """
    Xw = []
    yw = []
    for i in range(X_epochs.shape[0]):
        trial = X_epochs[i]               # (chans, samples)
        lab = labels[i]
        n = trial.shape[-1]
        start = 0
        while start + window_size <= n:
            w = trial[:, start:start+window_size]
            Xw.append(w)
            yw.append(lab)
            start += step_size
    Xw = np.asarray(Xw)
    yw = np.asarray(yw)
    # add channel-last singleton for EEGNet: (n_windows, chans, samples, 1)
    Xw = Xw[..., np.newaxis]
    return Xw, yw

# -----------------------
# MNE setup
# -----------------------
mne.set_config('MNE_DATA', MNE_DATA_PATH)
montage = make_standard_montage("standard_1005")  # reused across subjects

# Storage accumulators
train_X_list = []
train_y_list = []
test_X = None
test_y = None

# -----------------------
# Process each subject
# -----------------------
for subj in SUBJECTS:
    print(f"\n=== Loading subject {subj} ===")
    try:
        raw_fnames = eegbci.load_data(subj, RUNS)
    except Exception as e:
        print(f"  failed to load subject {subj}: {e}; skipping")
        continue

    raws = [read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = concatenate_raws(raws)

    # standardize / montage / ref / filter / resample
    eegbci.standardize(raw)
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    raw.filter(7., 30., fir_design="firwin", skip_by_annotation="edge")
    raw.resample(RESAMPLE_SFREQ, npad="auto")

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

    # get annotation mapping and find hands/feet keys
    events, annot_map = mne.events_from_annotations(raw)
    print("  annot_map sample:", {k: annot_map[k] for k in list(annot_map)[:4]})
    hands_key, feet_key = find_hand_feet_keys(annot_map)
    if hands_key is None or feet_key is None:
        print(f"  Could not find hands/feet keys for subject {subj}, annot_map={annot_map}; skipping subject.")
        continue

    event_id = {'hands': annot_map[hands_key], 'feet': annot_map[feet_key]}
    print(f"  Using event_id {event_id}")

    # build epochs (full trial window)
    epochs = Epochs(raw, event_id=event_id, tmin=TMIN, tmax=TMAX, proj=True, picks=picks, baseline=None, preload=True)
    print(f"  Num epochs: {len(epochs)}  epoch shape (chans,samples): {epochs.get_data().shape[1:]}")

    # map event codes -> 0/1
    code_to_label = {event_id['hands']: 0, event_id['feet']: 1}
    epoch_codes = epochs.events[:, -1]
    labels = np.array([code_to_label[c] for c in epoch_codes])
    print("  label counts:", np.bincount(labels))

    X_epochs = epochs.get_data() * 1000.0  
    Xw, yw = sliding_windows_from_epochs(X_epochs, labels, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    print(f"  produced windows: {Xw.shape}, classes: {np.bincount(yw)}")

    # append to train or test depending on subject
    if subj == TEST_SUBJECT:
        test_X = Xw
        test_y = yw
        print(f"  -> assigned to TEST (subject {subj})")
    else:
        train_X_list.append(Xw)
        train_y_list.append(yw)
        print(f"  -> appended to TRAIN pool")

# -----------------------
# Concatenate train data
# -----------------------
if len(train_X_list) == 0:
    raise RuntimeError("No training subjects processed - check SUBJECTS list and files.")

X_train = np.concatenate(train_X_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)

print("\nFinal dataset sizes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape, "test_X:", None if test_X is None else test_X.shape)

# Quick sanity checks
assert set(np.unique(y_train)).issubset({0,1})
if test_X is None:
    raise RuntimeError("Test subject produced no windows; check TEST_SUBJECT and its data.")

# -----------------------
# Balanced subsampling (optional)
# -----------------------
# If huge imbalance across classes across subjects, consider downsampling majority class or using class_weight.
print("Class balance train:", np.bincount(y_train), "test:", np.bincount(test_y))

# -----------------------
# Prepare labels (one-hot)
# -----------------------
nb_classes = 2
y_train_hot = np_utils.to_categorical(y_train, nb_classes)
y_test_hot = np_utils.to_categorical(test_y, nb_classes)

# -----------------------
# Build EEGNet
# -----------------------
chans = X_train.shape[1]
samples = X_train.shape[2]
print("EEGNet input dims -> chans:", chans, "samples:", samples)

model = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples,
               dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout')

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print("Model params:", model.count_params())

# callbacks
checkpointer = ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_best_only=True)

# small validation split from train for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_hot, test_size=0.15, random_state=SEED, stratify=y_train)

# -----------------------
# Fit
# -----------------------
history = model.fit(X_tr, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2,
                    validation_data=(X_val, y_val), callbacks=[checkpointer])

# -----------------------
# Evaluate on test subject
# -----------------------
model.load_weights(CHECKPOINT_PATH)
probs = model.predict(test_X)
preds = probs.argmax(axis=1)
acc = (preds == test_y).mean()

model.save("crossSub_model.h5")

print(f"\nCross-subject test accuracy on subject {TEST_SUBJECT}: {acc:.3f}")
print("Confusion matrix (test):")
cm = confusion_matrix(test_y, preds, labels=[0,1])
disp = ConfusionMatrixDisplay(cm, display_labels=['hands','feet'])
disp.plot()
plt.show()

# -----------------------
# Important sanity checks / next steps
# -----------------------
# 1) If test accuracy is near chance, try training on more subjects, or consider domain adaptation.
# 2) To avoid memory issues, change per-subject window saving to disk:
#    - For each subject, save Xw,yw to 'subject_{subj}_windows.npz' and later load and concat.
# 3) To evaluate robustness: do k-fold leave-subjects-out or test on multiple held-out subjects.
