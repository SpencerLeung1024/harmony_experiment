"""
Tuning Demo GUI

A simple GUI to explore how different tuning systems affect note frequencies.
Enter key indices (0-127, space-separated) to see the note names and frequencies
in various tuning systems, and play them on a piano.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd

from tuning_systems import get_tuning_system
from instruments import get_instrument

# Constants from main.py
PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
SAMPLE_RATE = 22050
NOTE_DURATION = 1.0  # seconds for each note when playing


def midi_to_note_name(midi_number: int) -> str:
    """Convert MIDI note number to note name (e.g., 60 -> 'C4')."""
    pitch_class_index = midi_number % 12
    octave = (midi_number // 12) - 1
    pitch_class = PITCH_CLASSES[pitch_class_index]
    return f"{pitch_class}{octave}"


def note_name_to_midi(note_name: str) -> int:
    """Convert note name to MIDI note number (e.g., 'C4' -> 60)."""
    # Handle octave -1
    if note_name[-2] == '-':
        pitch_class = note_name[:-2]
        octave = -1
    else:
        pitch_class = note_name[:-1]
        octave = int(note_name[-1])
    
    pitch_class_index = PITCH_CLASSES.index(pitch_class)
    midi_number = (octave + 1) * 12 + pitch_class_index
    return midi_number


class TuningDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tuning System Demo")
        self.root.geometry("700x500")
        
        # Initialize tuning systems
        self.tuning_systems = {
            "12-TET": get_tuning_system("12-TET"),
            "Pythagorean": get_tuning_system("Pythagorean"),
            "1/4-comma Meantone": get_tuning_system("1/4-comma Meantone"),
            "1/3-comma Meantone": get_tuning_system("1/3-comma Meantone"),
        }
        
        # Initialize piano instrument
        self.piano = get_instrument("piano")
        
        self.current_keys = []
        
        self._create_widgets()
    
    def _create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Input section
        ttk.Label(main_frame, text="Key Indices:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        
        self.key_input = ttk.Entry(main_frame, width=50)
        self.key_input.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=5)
        self.key_input.insert(0, "60 64 67")  # Default: C4 E4 G4
        
        ttk.Button(main_frame, text="Update", command=self._update_display).grid(
            row=0, column=2, pady=5
        )
        
        # Help text
        ttk.Label(main_frame, text="Enter space-separated key indices (0-127)", 
                 font=('Arial', 8), foreground='gray').grid(
            row=1, column=1, sticky=tk.W
        )
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        
        # Note names display
        ttk.Label(main_frame, text="Notes:", font=('Arial', 10, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.notes_label = ttk.Label(main_frame, text="", font=('Arial', 12))
        self.notes_label.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=5)
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        
        # Frequencies section header
        ttk.Label(main_frame, text="Frequencies (Hz):", font=('Arial', 10, 'bold')).grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=5
        )
        
        # Create a frame for the tuning system displays
        self.tuning_frame = ttk.Frame(main_frame)
        self.tuning_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.tuning_frame.columnconfigure(1, weight=1)
        
        # Create rows for each tuning system
        self.tuning_labels = {}
        self.freq_labels = {}
        
        for i, (name, tuning) in enumerate(self.tuning_systems.items()):
            row = i
            
            # Tuning system name
            ttk.Label(self.tuning_frame, text=f"{name}:", font=('Arial', 10)).grid(
                row=row, column=0, sticky=tk.W, pady=3, padx=(0, 10)
            )
            
            # Frequency display
            freq_label = ttk.Label(self.tuning_frame, text="", font=('Courier', 10))
            freq_label.grid(row=row, column=1, sticky=tk.W, pady=3)
            self.freq_labels[name] = freq_label
            
            # Play button
            play_btn = ttk.Button(
                self.tuning_frame, 
                text="▶ Play",
                command=lambda n=name: self._play_chord(n)
            )
            play_btn.grid(row=row, column=2, pady=3, padx=(10, 0))
        
        # Separator
        ttk.Separator(main_frame, orient='horizontal').grid(
            row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10
        )
        
        # Comparison section
        ttk.Label(main_frame, text="Comparison:", font=('Arial', 10, 'bold')).grid(
            row=8, column=0, sticky=tk.W, pady=5
        )
        
        self.comparison_text = tk.Text(main_frame, height=8, width=60, font=('Courier', 9))
        self.comparison_text.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Add scrollbar to comparison text
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=self.comparison_text.yview)
        scrollbar.grid(row=9, column=3, sticky=(tk.N, tk.S))
        self.comparison_text['yscrollcommand'] = scrollbar.set
        
        # Bind Enter key to update
        self.key_input.bind('<Return>', lambda e: self._update_display())
        
        # Initial update
        self._update_display()
    
    def _parse_keys(self) -> list:
        """Parse the key indices from the input field."""
        try:
            key_str = self.key_input.get().strip()
            if not key_str:
                return []
            keys = [int(k) for k in key_str.split()]
            # Validate range
            for k in keys:
                if k < 0 or k > 127:
                    raise ValueError(f"Key {k} out of range (0-127)")
            return keys
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid key indices (0-127): {e}")
            return []
    
    def _update_display(self):
        """Update all displays based on current key indices."""
        self.current_keys = self._parse_keys()
        if not self.current_keys:
            return
        
        # Update note names
        note_names = [midi_to_note_name(k) for k in self.current_keys]
        self.notes_label.config(text="  ".join(note_names))
        
        # Update frequency displays for each tuning system
        for name, tuning in self.tuning_systems.items():
            freqs = []
            for key in self.current_keys:
                freq = tuning.key_to_freq(key)
                if freq is not None:
                    freqs.append(f"{freq:8.2f}")
                else:
                    freqs.append("   N/A  ")
            self.freq_labels[name].config(text="  ".join(freqs))
        
        # Update comparison text
        self._update_comparison()
    
    def _update_comparison(self):
        """Update the comparison text showing differences between tunings."""
        self.comparison_text.delete('1.0', tk.END)
        
        if len(self.current_keys) == 0:
            return
        
        # Calculate cents deviation from 12-TET for each note
        tet_tuning = self.tuning_systems["12-TET"]
        
        lines = []
        lines.append(f"{'Note':<6} {'Key':<5} {'12-TET (Hz)':<12} " + 
                    f"{'Pyth (¢)':<10} {'1/4MT (¢)':<10} {'1/3MT (¢)':<10}")
        lines.append("-" * 65)
        
        for key in self.current_keys:
            note_name = midi_to_note_name(key)
            tet_freq = tet_tuning.key_to_freq(key)
            
            line = f"{note_name:<6} {key:<5} {tet_freq:<12.2f} "
            
            for tuning_name in ["Pythagorean", "1/4-comma Meantone", "1/3-comma Meantone"]:
                tuning = self.tuning_systems[tuning_name]
                freq = tuning.key_to_freq(key)
                if freq is not None and tet_freq is not None:
                    # Calculate cents deviation
                    cents = 1200 * np.log2(freq / tet_freq)
                    line += f"{cents:>+8.1f}  "
                else:
                    line += f"{'N/A':<10}"
            
            lines.append(line)
        
        # Add interval comparison
        if len(self.current_keys) >= 2:
            lines.append("")
            lines.append("Interval Comparisons (cents from 12-TET):")
            lines.append("-" * 65)
            
            for i in range(len(self.current_keys) - 1):
                key1 = self.current_keys[i]
                key2 = self.current_keys[i + 1]
                note1 = midi_to_note_name(key1)
                note2 = midi_to_note_name(key2)
                
                lines.append(f"{note1} -> {note2}:")
                
                for tuning_name in ["Pythagorean", "1/4-comma Meantone", "1/3-comma Meantone"]:
                    tuning = self.tuning_systems[tuning_name]
                    tet = self.tuning_systems["12-TET"]
                    
                    freq1 = tuning.key_to_freq(key1)
                    freq2 = tuning.key_to_freq(key2)
                    tet_freq1 = tet.key_to_freq(key1)
                    tet_freq2 = tet.key_to_freq(key2)
                    
                    if all(f is not None for f in [freq1, freq2, tet_freq1, tet_freq2]):
                        # Interval in this tuning
                        interval_cents = 1200 * np.log2(freq2 / freq1)
                        # Interval in 12-TET
                        tet_interval_cents = 1200 * np.log2(tet_freq2 / tet_freq1)
                        # Deviation
                        deviation = interval_cents - tet_interval_cents
                        
                        lines.append(f"  {tuning_name:<20} Interval: {interval_cents:>7.1f}¢  (dev: {deviation:>+6.1f}¢)")
        
        self.comparison_text.insert('1.0', "\n".join(lines))
    
    def _play_chord(self, tuning_name: str):
        """Play the current chord using the specified tuning system."""
        if not self.current_keys:
            return
        
        tuning = self.tuning_systems[tuning_name]
        
        # Get frequencies for each key
        frequencies = []
        for key in self.current_keys:
            freq = tuning.key_to_freq(key)
            if freq is not None:
                frequencies.append(freq)
        
        if not frequencies:
            messagebox.showwarning("Cannot Play", "No valid frequencies for these keys.")
            return
        
        # Generate audio for each note and mix them
        total_samples = int(NOTE_DURATION * SAMPLE_RATE) + int(0.5 * SAMPLE_RATE)  # Add release time
        mixed_audio = np.zeros(total_samples)
        
        for freq in frequencies:
            sound = self.piano.get_sound(freq, 0.5, NOTE_DURATION, SAMPLE_RATE)
            # Ensure same length
            if len(sound) < total_samples:
                sound = np.pad(sound, (0, total_samples - len(sound)), mode='constant')
            elif len(sound) > total_samples:
                sound = sound[:total_samples]
            mixed_audio += sound
        
        # Normalize to prevent clipping
        max_amp = np.max(np.abs(mixed_audio))
        if max_amp > 0:
            mixed_audio = mixed_audio / max_amp * 0.5
        
        # Play the audio
        try:
            sd.play(mixed_audio, SAMPLE_RATE)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {e}")


def main():
    root = tk.Tk()
    app = TuningDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
