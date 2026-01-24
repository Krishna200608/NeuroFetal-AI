import os
import glob
import numpy as np
import wfdb
import pandas as pd
from scipy.interpolate import interp1d

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "ctu_uhb_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "Datasets", "processed")
TARGET_FS = 1  # 1 Hz
DURATION_MIN = 60
DURATION_SEC = DURATION_MIN * 60
GAP_THRESHOLD_SEC = 15

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_header(header_path):
    """
    Parses the .hea file to extract Age, Parity, Gestation, pH.
    Values are typically in comments: '#Age 32', '#pH 7.14', '#Gest. weeks 37'
    """
    features = {
        'Age': None,
        'Parity': None,
        'Gestation': None, # Gest. weeks
        'pH': None
    }
    
    with open(header_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Remove '#' and leading/trailing whitespace
                content = line[1:].strip()
                
                # Check for specific keys
                # We use endswith or split()[-1] to get the value, as the value is usually the last item
                # But be careful if there are units or comments. 
                # In CTU-UHB .hea: "#Age 32", "#Gest. weeks 37"
                
                parts = content.split()
                if not parts:
                    continue
                    
                val_str = parts[-1]
                
                if content.startswith('Age'):
                    try: features['Age'] = float(val_str)
                    except: pass
                
                elif content.startswith('Parity'):
                    try: features['Parity'] = float(val_str)
                    except: pass
                    
                elif content.startswith('Gest. weeks'):
                    try: features['Gestation'] = float(val_str)
                    except: pass
                    
                elif content.startswith('pH'):
                    try: features['pH'] = float(val_str)
                    except: pass
                        
    return features
                        
    return features

def process_signal(signal, fs):
    """
    Process FHR signal:
    1. Check for 0s (missing).
    2. Interpolate gaps < 15s.
    3. Keep gaps > 15s as 0.
    4. Crop last 60 mins.
    5. Resample to 1Hz.
    6. Normalize (MinMax).
    """
    # Create time array
    n_samples = len(signal)
    t = np.arange(n_samples) / fs
    
    # Identify gaps (0 values)
    # create a mask where 0 is True
    is_gap = (signal == 0)
    
    # We need to find segments of gaps
    # A simple way: use pandas for grouping or manually iterate
    # Let's use a manual iteration for clear logic
    
    processed_signal = signal.copy()
    
    # Find continuous runs of zeros
    # make sure to handle 0s at start/end
    
    # Convert to boolean array
    mask = is_gap.astype(int)
    
    # Find changes in the mask
    diff = np.diff(np.concatenate(([0], mask, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        gap_len_samples = end - start
        gap_len_sec = gap_len_samples / fs
        
        if gap_len_sec < GAP_THRESHOLD_SEC:
            # Interpolate
            # Points to interpolate between are start-1 and end
            # Handle edge cases
            left_idx = start - 1
            right_idx = end
            
            if left_idx < 0 or right_idx >= n_samples:
                # Cannot interpolate at boundaries if signal missing
                continue
                
            x_known = [left_idx, right_idx]
            y_known = [signal[left_idx], signal[right_idx]]
            
            # Linear interpolation
            interp_func = interp1d(x_known, y_known, kind='linear')
            processed_signal[start:end] = interp_func(np.arange(start, end))
            
    # Crop last 60 minutes
    # Target length in samples
    required_samples = DURATION_SEC * fs
    
    if len(processed_signal) < required_samples:
        # Pad with zeros if too short (though unlikely for this dataset usually)
        pad_len = int(required_samples - len(processed_signal))
        processed_signal = np.pad(processed_signal, (pad_len, 0), 'constant')
    else:
        processed_signal = processed_signal[-int(required_samples):]
        
    # Resample to 1Hz
    # Current fs = 4Hz usually. Target 1Hz.
    # We decimate or interpolate. Dwnsampling by integer factor if possible.
    # 4Hz -> 1Hz is factor of 4.
    
    num_samples_target = DURATION_SEC * TARGET_FS
    
    # Resample. signal.resample might introduce artifacts for 0s?
    # Better to just slice if integer factor? 
    # But usually resampling with polyphase filter is better (scipy.signal.resample)
    # However, for FHR, simple averaging or decimation might be safer to preserve 0s?
    # User says "Resample". Let's use interpolation to exact target grid.
    
    x_old = np.linspace(0, DURATION_SEC, len(processed_signal))
    x_new = np.linspace(0, DURATION_SEC, num_samples_target)
    
    # Use nearest or linear? Linear is better for continuous signal.
    # But 0s (gaps) might be smeared.
    # If gap > 15s it is 0. 
    # Should we ignore 0s during resampling? 
    # Paper usually implies standard resampling.
    
    f_resample = interp1d(x_old, processed_signal, kind='linear', bounds_error=False, fill_value=0)
    resampled_signal = f_resample(x_new)
    
    # Normalize MinMax (0-1)
    # Ignorning 0s for min/max calculation? The prompt says "Normalize... to the signal".
    # Usually MinMax on the valid range.
    # If 0 is "missing", it shouldn't bias the min.
    # However, user validation says "replace with 0". So 0 is part of the signal now.
    # But FHR typically ranges 50-200. 0 is outlier.
    # Standard practice: Normalize non-zero values, then set 0s back to 0 (or keep -1).
    # Prompt says "MinMax Scaling (0-1)".
    # If I include 0, the range is 0-200, so valid FHR(120) becomes 0.6.
    # If I exclude 0, range 50-200, valid FHR(120) becomes (120-50)/150 = 0.46. And 0 becomes -0.33?
    # Re-reading prompt: "Normalization: Apply MinMax Scaling (0-1) to the signal."
    # AND "Dataset... Fetal Heart Rate Analysis".
    # Given the strict constraint to the paper, papers usually min-max scale the valid range (e.g. 50-220 bpm) to 0-1.
    # Or just min-max of the array.
    # Let's check min/max of the array.
    
    _min = np.min(resampled_signal)
    _max = np.max(resampled_signal)
    
    # Refinement: If 0 denotes missing, strict MinMax on full array makes 0 the min.
    # Most likely this is intended if the input to CNN is just the raw values.
    # I will use the array min/max.
    if _max > _min:
        resampled_signal = (resampled_signal - _min) / (_max - _min)
    else:
        resampled_signal = np.zeros_like(resampled_signal)
        
    return resampled_signal

def main():
    ensure_dir(PROCESSED_DATA_DIR)
    
    record_paths = glob.glob(os.path.join(RAW_DATA_DIR, "*.hea"))
    print(f"Found {len(record_paths)} records.")
    
    X_fhr = []
    X_tabular = []
    y = []
    
    cnt = 0
    valid_cnt = 0
    
    for hea_path in record_paths:
        base = os.path.splitext(hea_path)[0]
        rec_name = os.path.basename(base)
        
        # Parse Header
        feats = parse_header(hea_path)
        
        # Check if we have all tabular features
        # If any is NaN, we might have to drop or impute.
        # For now, let's collect and see.
        # But 'is_compromised' depends on pH.
        if feats['pH'] is None or np.isnan(feats['pH']):
            print(f"Skipping {rec_name}: Missing pH")
            continue
            
        # Target
        is_compromised = 1 if feats['pH'] < 7.05 else 0
        
        # Read Signal
        try:
            # wfdb.rdsamp returns signals, fields
            signals, fields = wfdb.rdsamp(base)
            # FHR is usually channel 0 (based on my view of 1001.hea line 2)
            # 1001.dat 16 100(0)/bpm ... FHR
            # Yes, standard CTU-UHB has FHR at index 0, UC at index 1.
            fhr_signal = signals[:, 0]
            fs = fields['fs']
            
            # Preprocess Signal
            processed_fhr = process_signal(fhr_signal, fs)
            
            # Prepare Tabular Vector
            # Age, Parity, Gestation, pH (pH is target, but maybe feature too? NO, pH is output validation, usually not input)
            # Prompt: "Extract 'Age', 'Parity', 'Gestation', and 'pH'. Target Label: ... is_compromised if pH < 7.05"
            # Branch 2 Input: "Tabular... (N_features,)". Usually pH is NOT an input feature if it defines the label.
            # I will exclude pH from X_tabular.
            
            # Need to handle missing tabular inputs?
            # Assuming linear fill or mean later. Here just raw.
            tab_vec = [feats['Age'], feats['Parity'], feats['Gestation']]
            
            X_fhr.append(processed_fhr)
            X_tabular.append(tab_vec)
            y.append(is_compromised)
            
            valid_cnt += 1
            
        except Exception as e:
            print(f"Error processing {rec_name}: {e}")
            
        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} files...")
            
    # Convert to arrays
    X_fhr = np.array(X_fhr)
    X_tabular = np.array(X_tabular)
    y = np.array(y)
    
    # Handle NaNs in tabular data?
    # Simple imputation: replace nan with mean
    # Need to warn user if doing this.
    col_means = np.nanmean(X_tabular, axis=0)
    inds = np.where(np.isnan(X_tabular))
    X_tabular[inds] = np.take(col_means, inds[1])
    
    # Save
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_fhr.npy"), X_fhr)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_tabular.npy"), X_tabular)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y.npy"), y)
    
    print(f"Processing complete. Saved {valid_cnt} records.")
    print(f"Shapes: X_fhr {X_fhr.shape}, X_tabular {X_tabular.shape}, y {y.shape}")
    print(f"Class balance: {np.sum(y)} compromised / {len(y)} total")

if __name__ == "__main__":
    main()
