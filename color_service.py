import numpy as np
from matplotlib import pyplot as plt

from members import Member

MIDDLE_C = 261.626 # Hz, MIDI key 60

def bin_to_hz(bin: int, num_bins: int, sample_rate: int) -> float:
    return bin * sample_rate / (2 * num_bins)

def hz_to_bin(freq: float, num_bins: int, sample_rate: int) -> int:
    return int((freq / (sample_rate / 2)) * num_bins)

def freq_to_hue(freq: float) -> float:
    # C (in any octave) is red
    # Hue increases with pitch class, wrapping around at the octave
    # Hue is in [0, 1)
    ratio = freq / MIDDLE_C
    hue = np.log2(ratio) % 1
    return hue

class ColorService:
    @staticmethod
    def color_weights(
        member: Member,
    ):
        (num_keys, num_notes) = member.weights.shape
        tuning_system = member.tuning_system
        instrument_range_low = member.instrument_range[0]

        color_stick = np.zeros((num_keys, 3))
        for key in range(num_keys):
            freq = tuning_system.key_to_freq(instrument_range_low + key)
            hue = freq_to_hue(freq)
            color_stick[key] = plt.cm.hsv(hue)[:3]
        
        # Element-wise multiply weights with color stick
        colormap = np.reshape(member.weights.detach().numpy(), (num_keys, num_notes, 1)) * np.reshape(color_stick, (num_keys, 1, 3))
        return colormap
    
    @staticmethod
    def color_activations(
        member: Member,
    ):
        # Basically the same thing but use activations (argmax for MonophonicMember)
        (num_keys, num_notes) = member.weights.shape
        tuning_system = member.tuning_system
        instrument_range_low = member.instrument_range[0]

        color_stick = np.zeros((num_keys, 3))
        for key in range(num_keys):
            freq = tuning_system.key_to_freq(instrument_range_low + key)
            hue = freq_to_hue(freq)
            color_stick[key] = plt.cm.hsv(hue)[:3]
        
        # Element-wise multiply activations with color stick
        colormap = np.reshape(member.forward(None).detach().numpy(), (num_keys, num_notes, 1)) * np.reshape(color_stick, (num_keys, 1, 3))
        return colormap
    
    @staticmethod
    def color_spectrogram(
        spectrogram: np.ndarray,
        sample_rate: int
    ):
        (num_bins, num_frames) = spectrogram.shape
        color_stick = np.zeros((num_bins, 3))
        for bin in range(num_bins):
            # The first bin is 0 Hz which doesn't map to a key, so just leave it black
            if bin == 0:
                continue
            freq = bin_to_hz(bin, num_bins, sample_rate)
            hue = freq_to_hue(freq)
            color_stick[bin] = plt.cm.hsv(hue)[:3]
        
        # Amplitudes are all over the place. We need [0.0, 1.0]
        # But a massive amount of energy is in infrasound that we don't have the resolution to look at anyways
        # So find max in "interesting" part (50 Hz to 10 kHz) and use that as our scaling factor
        interesting_bin_low = hz_to_bin(50, num_bins, sample_rate)
        interesting_bin_high = hz_to_bin(10000, num_bins, sample_rate)
        max_in_interesting_part = spectrogram[interesting_bin_low:interesting_bin_high, :].max()
        #max_overall = spectrogram.max()
        #print(f"Spectrogram max overall: {max_overall:.4f}, max in 50Hz-10kHz: {max_in_interesting_part:.4f}")
        # Spectrogram max overall: 4572.6846, max in 50Hz-10kHz: 2770.6194
        spectrogram = spectrogram / max_in_interesting_part
        
        # Element-wise multiply spectrogram with color stick
        colormap = np.reshape(spectrogram, (num_bins, num_frames, 1)) * np.reshape(color_stick, (num_bins, 1, 3))
        return colormap
