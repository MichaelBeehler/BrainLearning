import matplotlib
# Use Agg backend for non-interactive plotting (saves to files)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import CSP
from mne.time_frequency import psd_array_welch

troubleshoot_epochs = False
save_plots = True  # Set to False if you want interactive plots (requires GUI backend)

# Try to set browser backend, but fallback gracefully if not available
try:
    mne.viz.set_browser_backend("matplotlib")  # Use matplotlib backend instead of qt
except:
    print("Note: Using default MNE browser backend")
raw = mne.io.read_raw_edf("S001R03.edf", preload=True)

print(raw.info)

raw.filter(l_freq=1.0, h_freq=40.0)  # 1–40 Hz bandpass
raw.notch_filter(freqs=[60])

# Skip interactive plots if save_plots is True (they require GUI)
if not save_plots:
    raw.plot(n_channels=10, scalings='auto')
    plt.show()

# Plot PSD for all channels
fig1 = raw.plot_psd(fmin=1, fmax=50, average=True, show=False)
if save_plots:
    fig1.savefig('eeg_psd_linear.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("Saved: eeg_psd_linear.png")
else:
    plt.show()

# Or log-scaled with more detail
fig2 = raw.plot_psd(fmin=1, fmax=50, average=True, dB=True, show=False)
if save_plots:
    fig2.savefig('eeg_psd_log.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("Saved: eeg_psd_log.png")
else:
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

print(f"Raw data shape: {X.shape}, Labels shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_band_power_features(epochs, bands=None):
    """
    Extract band power features from epochs.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Preprocessed epochs object
    bands : dict
        Dictionary of frequency bands, e.g., {'delta': (1, 4), 'theta': (4, 8),
        'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 40)}
    
    Returns:
    --------
    features : np.array
        Feature matrix of shape (n_epochs, n_channels * n_bands)
    band_names : list
        List of feature names
    """
    if bands is None:
        # Standard frequency bands for motor imagery
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'mu': (8, 12),      # Mu band - crucial for motor imagery
            'alpha': (8, 13),
            'beta': (13, 30),   # Beta band - important for motor imagery
            'gamma': (30, 40)
        }
    
    # Get sampling frequency
    sfreq = epochs.info['sfreq']
    
    # Extract data
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # Compute PSD using Welch's method
    # Use overlapping windows for better frequency resolution
    psds, freqs = psd_array_welch(data, sfreq, fmin=1, fmax=40, 
                                   n_fft=int(sfreq * 2), n_overlap=int(sfreq))
    # psds shape: (n_epochs, n_channels, n_freqs)
    
    # Extract power in each band
    features_list = []
    band_names = []
    
    for band_name, (fmin, fmax) in bands.items():
        # Find frequencies in this band
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        if not freq_mask.any():
            continue
            
        # Average power across frequencies in this band (mean or log)
        band_power = np.mean(psds[:, :, freq_mask], axis=2)
        # band_power shape: (n_epochs, n_channels)
        
        # Log transform for better distribution (optional but common)
        band_power_log = np.log10(band_power + 1e-10)
        
        features_list.append(band_power_log)
        
        # Create feature names
        for ch_idx in range(n_channels):
            ch_name = epochs.ch_names[ch_idx] if epochs.ch_names else f'ch{ch_idx}'
            band_names.append(f'{ch_name}_{band_name}_log_power')
    
    # Concatenate all bands
    features = np.concatenate(features_list, axis=1)  # (n_epochs, n_channels * n_bands)
    
    return features, band_names


def extract_csp_features(epochs, n_components=4):
    """
    Extract Common Spatial Patterns (CSP) features.
    CSP is very effective for motor imagery classification.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Preprocessed epochs object with class labels
    n_components : int
        Number of CSP components to extract
    
    Returns:
    --------
    features : np.array
        Feature matrix of shape (n_epochs, n_components)
    csp : CSP object
        Fitted CSP transformer
    """
    # CSP requires at least 2 classes
    unique_labels = np.unique(epochs.events[:, -1])
    if len(unique_labels) < 2:
        print("Warning: CSP requires at least 2 classes. Returning empty features.")
        return np.empty((len(epochs), n_components)), None
    
    # Create CSP transformer
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    
    # Fit and transform
    # CSP expects data shape: (n_epochs, n_channels, n_times)
    X_csp = epochs.get_data()
    y_labels = epochs.events[:, -1]
    
    # Fit CSP on training data (uses covariance matrices)
    csp.fit(X_csp, y_labels)
    
    # Transform
    features = csp.transform(X_csp)
    
    return features, csp


def extract_combined_features(epochs, use_csp=True, use_band_power=True, 
                              csp_components=4, bands=None):
    """
    Extract combined features: band power + CSP.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        Preprocessed epochs object
    use_csp : bool
        Whether to include CSP features
    use_band_power : bool
        Whether to include band power features
    csp_components : int
        Number of CSP components
    bands : dict
        Frequency bands for band power extraction
    
    Returns:
    --------
    features : np.array
        Combined feature matrix
    feature_info : dict
        Dictionary with feature names and extraction details
    csp_object : CSP object or None
        Fitted CSP transformer for visualization
    """
    features_list = []
    csp_object = None
    feature_info = {
        'band_power': None,
        'csp': None,
        'total_features': 0
    }
    
    # Extract band power features
    if use_band_power:
        print("Extracting band power features...")
        band_features, band_names = extract_band_power_features(epochs, bands=bands)
        features_list.append(band_features)
        feature_info['band_power'] = {
            'shape': band_features.shape,
            'names': band_names[:10]  # Store first 10 names as sample
        }
        print(f"  Band power features: {band_features.shape}")
    
    # Extract CSP features
    if use_csp:
        print("Extracting CSP features...")
        csp_features, csp = extract_csp_features(epochs, n_components=csp_components)
        if csp is not None:
            features_list.append(csp_features)
            csp_object = csp  # Store CSP object
            feature_info['csp'] = {
                'shape': csp_features.shape,
                'n_components': csp_components
            }
            print(f"  CSP features: {csp_features.shape}")
    
    # Combine features
    if features_list:
        features = np.concatenate(features_list, axis=1)
        feature_info['total_features'] = features.shape[1]
        print(f"\nTotal combined features: {features.shape}")
        return features, feature_info, csp_object
    else:
        print("No features extracted!")
        return None, feature_info, None


# Extract features
print("\n" + "="*60)
print("FEATURE EXTRACTION")
print("="*60)

# Extract combined features
X_features, feature_info, csp_object = extract_combined_features(
    epochs, 
    use_csp=True, 
    use_band_power=True,
    csp_components=6,  # Number of CSP components to extract
    bands={
        'mu': (8, 12),      # Mu rhythm - key for motor imagery
        'beta': (13, 30),   # Beta band - also important
    }
)

if X_features is not None:
    print(f"\nFinal feature matrix shape: {X_features.shape}")
    print(f"Feature info: {feature_info}")
    print(f"Features per epoch: {X_features.shape[1]}")


# ============================================================================
# FEATURE VISUALIZATION
# ============================================================================

def visualize_features(X_features, y, feature_info, epochs):
    """
    Create comprehensive visualizations of extracted features.
    
    Parameters:
    -----------
    X_features : np.array
        Feature matrix (n_epochs, n_features)
    y : np.array
        Class labels
    feature_info : dict
        Dictionary with feature information
    epochs : mne.Epochs
        Epochs object for additional context
    """
    unique_classes = np.unique(y)
    class_names = {0: 'Rest', 1: 'Motor Imagery'}
    n_classes = len(unique_classes)
    
    # Separate features by type
    band_power_features = None
    csp_features = None
    start_idx = 0
    
    if feature_info.get('band_power') is not None:
        bp_shape = feature_info['band_power']['shape']
        n_bp_features = bp_shape[1]
        band_power_features = X_features[:, start_idx:start_idx + n_bp_features]
        start_idx += n_bp_features
        print(f"Band power features extracted: {band_power_features.shape}")
    
    if feature_info.get('csp') is not None:
        csp_shape = feature_info['csp']['shape']
        n_csp_features = csp_shape[1]
        csp_features = X_features[:, start_idx:start_idx + n_csp_features]
        print(f"CSP features extracted: {csp_features.shape}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Feature distribution by class
    ax1 = plt.subplot(3, 3, 1)
    for class_label in unique_classes:
        class_mask = y == class_label
        class_features = X_features[class_mask]
        mean_features = np.mean(class_features, axis=0)
        std_features = np.std(class_features, axis=0)
        ax1.errorbar(range(len(mean_features)), mean_features, yerr=std_features, 
                     label=class_names[class_label], alpha=0.7, linewidth=1)
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Value')
    ax1.set_title('Mean Feature Values by Class')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Feature distribution histogram
    ax2 = plt.subplot(3, 3, 2)
    for class_label in unique_classes:
        class_mask = y == class_label
        # Use a representative feature (median feature)
        median_feat_idx = X_features.shape[1] // 2
        class_features = X_features[class_mask, median_feat_idx]
        ax2.hist(class_features, bins=20, alpha=0.6, label=class_names[class_label], density=True)
    ax2.set_xlabel(f'Feature Value (Feature #{median_feat_idx})')
    ax2.set_ylabel('Density')
    ax2.set_title('Feature Distribution Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Band power comparison (if available)
    if band_power_features is not None:
        ax3 = plt.subplot(3, 3, 3)
        n_channels = len(epochs.ch_names) if epochs.ch_names else band_power_features.shape[1] // 2
        n_bands = band_power_features.shape[1] // n_channels if n_channels > 0 else 1
        
        # Average band power across channels for each band
        if n_bands > 0:
            bp_reshaped = band_power_features.reshape((band_power_features.shape[0], n_channels, n_bands))
            bp_mean = np.mean(bp_reshaped, axis=1)  # Average across channels
            
            x_pos = np.arange(n_bands)
            width = 0.35
            
            for idx, class_label in enumerate(unique_classes):
                class_mask = y == class_label
                class_bp_mean = np.mean(bp_mean[class_mask], axis=0)
                offset = width * (idx - 0.5)
                ax3.bar(x_pos + offset, class_bp_mean, width, label=class_names[class_label], alpha=0.7)
            
            ax3.set_xlabel('Frequency Band')
            ax3.set_ylabel('Mean Log Power')
            ax3.set_title('Average Band Power by Class')
            band_labels = ['Mu', 'Beta'] if n_bands == 2 else [f'Band {i}' for i in range(n_bands)]
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(band_labels)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. CSP feature comparison (if available)
    if csp_features is not None:
        ax4 = plt.subplot(3, 3, 4)
        n_csp = csp_features.shape[1]
        x_pos = np.arange(n_csp)
        width = 0.35
        
        for idx, class_label in enumerate(unique_classes):
            class_mask = y == class_label
            class_csp_mean = np.mean(csp_features[class_mask], axis=0)
            offset = width * (idx - 0.5)
            ax4.bar(x_pos + offset, class_csp_mean, width, label=class_names[class_label], alpha=0.7)
        
        ax4.set_xlabel('CSP Component')
        ax4.set_ylabel('Mean Feature Value')
        ax4.set_title('CSP Features by Class')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'CSP {i+1}' for i in range(n_csp)])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Feature correlation matrix (first 20 features)
    ax5 = plt.subplot(3, 3, 5)
    n_features_to_plot = min(20, X_features.shape[1])
    corr_matrix = np.corrcoef(X_features[:, :n_features_to_plot].T)
    im = ax5.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax5.set_title(f'Feature Correlation (First {n_features_to_plot} Features)')
    ax5.set_xlabel('Feature Index')
    ax5.set_ylabel('Feature Index')
    plt.colorbar(im, ax=ax5)
    
    # 6. Box plot of features by class
    ax6 = plt.subplot(3, 3, 6)
    data_to_plot = [X_features[y == class_label, 0] for class_label in unique_classes]
    bp = ax6.boxplot(data_to_plot, labels=[class_names[c] for c in unique_classes], patch_artist=True)
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    ax6.set_ylabel('Feature Value')
    ax6.set_title('Feature Distribution (First Feature)')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. 2D scatter plot (first two CSP components or first two features)
    ax7 = plt.subplot(3, 3, 7)
    if csp_features is not None and csp_features.shape[1] >= 2:
        feat1 = csp_features[:, 0]
        feat2 = csp_features[:, 1]
        feat_label = 'CSP Components'
    else:
        feat1 = X_features[:, 0]
        feat2 = X_features[:, 1]
        feat_label = 'Features'
    
    for class_label in unique_classes:
        class_mask = y == class_label
        ax7.scatter(feat1[class_mask], feat2[class_mask], 
                   label=class_names[class_label], alpha=0.6, s=50)
    ax7.set_xlabel(f'{feat_label} 1')
    ax7.set_ylabel(f'{feat_label} 2')
    ax7.set_title('2D Feature Space')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Feature importance/variance
    ax8 = plt.subplot(3, 3, 8)
    feature_variance = np.var(X_features, axis=0)
    top_n = min(15, len(feature_variance))
    top_indices = np.argsort(feature_variance)[-top_n:]
    top_variances = feature_variance[top_indices]
    ax8.barh(range(top_n), top_variances)
    ax8.set_xlabel('Variance')
    ax8.set_ylabel('Feature Index')
    ax8.set_title(f'Top {top_n} Features by Variance')
    ax8.set_yticks(range(top_n))
    ax8.set_yticklabels([f'Feat {i}' for i in top_indices])
    ax8.grid(True, alpha=0.3, axis='x')
    
    # 9. Class separability (mean difference / std)
    ax9 = plt.subplot(3, 3, 9)
    if n_classes >= 2:
        class_0_mask = y == unique_classes[0]
        class_1_mask = y == unique_classes[1]
        
        mean_0 = np.mean(X_features[class_0_mask], axis=0)
        mean_1 = np.mean(X_features[class_1_mask], axis=0)
        std_0 = np.std(X_features[class_0_mask], axis=0)
        std_1 = np.std(X_features[class_1_mask], axis=0)
        
        # Compute separability metric
        separability = np.abs(mean_0 - mean_1) / (std_0 + std_1 + 1e-10)
        top_sep_n = min(15, len(separability))
        top_sep_indices = np.argsort(separability)[-top_sep_n:]
        top_separabilities = separability[top_sep_indices]
        
        ax9.barh(range(top_sep_n), top_separabilities, color='green', alpha=0.7)
        ax9.set_xlabel('Separability Score')
        ax9.set_ylabel('Feature Index')
        ax9.set_title(f'Top {top_sep_n} Most Discriminative Features')
        ax9.set_yticks(range(top_sep_n))
        ax9.set_yticklabels([f'Feat {i}' for i in top_sep_indices])
        ax9.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.suptitle('Feature Visualization Dashboard', y=1.02, fontsize=16, fontweight='bold')
    return fig


def visualize_csp_patterns(csp, epochs):
    """
    Visualize CSP spatial patterns if CSP was used.
    
    Parameters:
    -----------
    csp : CSP object
        Fitted CSP transformer
    epochs : mne.Epochs
        Epochs object for montage/channel info
    """
    if csp is None:
        return None
    
    try:
        # Get CSP patterns (spatial filters)
        csp_patterns = csp.patterns_
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        n_components = min(csp_patterns.shape[0], 6)
        
        for i in range(n_components):
            ax = axes[i]
            
            # Create a fake Evoked object for plotting
            # Use the first epoch's info as template
            ch_names = epochs.ch_names if epochs.ch_names else [f'CH{i}' for i in range(csp_patterns.shape[1])]
            
            # Try to plot as topomap if montage is available
            if epochs.info.get('montage') is not None:
                try:
                    im, _ = mne.viz.plot_topomap(
                        csp_patterns[i], epochs.info, axes=ax, 
                        show=False, cmap='RdBu_r'
                    )
                    ax.set_title(f'CSP Pattern {i+1}')
                except:
                    # Fallback to bar plot if topomap fails
                    ax.bar(range(len(csp_patterns[i])), csp_patterns[i])
                    ax.set_title(f'CSP Pattern {i+1}')
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Weight')
            else:
                # Bar plot if no montage
                ax.bar(range(len(csp_patterns[i])), csp_patterns[i], color='steelblue', alpha=0.7)
                ax.set_title(f'CSP Pattern {i+1}')
                ax.set_xlabel('Channel Index')
                ax.set_ylabel('Pattern Weight')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for i in range(n_components, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.suptitle('CSP Spatial Patterns', y=1.02, fontsize=16, fontweight='bold')
        return fig
    except Exception as e:
        print(f"Could not visualize CSP patterns: {e}")
        return None


# Visualize features
if X_features is not None:
    print("\n" + "="*60)
    print("CREATING FEATURE VISUALIZATIONS")
    print("="*60)
    
    # Main feature visualization
    fig1 = visualize_features(X_features, y, feature_info, epochs)
    if save_plots:
        fig1.savefig('feature_visualization_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("\nSaved: feature_visualization_dashboard.png")
    else:
        plt.show()
    
    # CSP patterns visualization
    if csp_object is not None:
        fig2 = visualize_csp_patterns(csp_object, epochs)
        if fig2 is not None:
            if save_plots:
                fig2.savefig('csp_patterns.png', dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print("Saved: csp_patterns.png")
            else:
                plt.show()




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