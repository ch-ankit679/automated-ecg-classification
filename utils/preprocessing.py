import pandas as pd
import numpy as np

def preprocess_ecg(file):
    df = pd.read_csv(file, header=None)

    # convert to float32
    signal = df.values.astype('float32')

    # if multiple rows, take first row
    if signal.shape[0] > 1:
        signal = signal[0]
    else:
        signal = signal.flatten()

    # FINAL SHAPE → (1, 1, length)
    signal = signal.reshape(1, 1, -1)

    return signal

def inject_gaussian_noise(signal, snr=15):
    """Injects synthetic high-frequency muscle artifact noise."""
    sig_1d = signal.flatten()
    signal_power = np.mean(sig_1d ** 2)
    noise_power = signal_power / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(sig_1d))
    noisy_sig = sig_1d + noise
    return noisy_sig.reshape(signal.shape).astype('float32')

def inject_baseline_wander(signal, freq=0.5, amp=0.5):
    """Injects low-frequency patient respiration movement noise."""
    sig_1d = signal.flatten()
    t = np.linspace(0, len(sig_1d)/360.0, len(sig_1d))
    wander = amp * np.sin(2 * np.pi * freq * t)
    noisy_sig = sig_1d + wander
    return noisy_sig.reshape(signal.shape).astype('float32')