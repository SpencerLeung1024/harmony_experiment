from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F
import numpy as np

from song import Song
from members import Member, PolyphonicMember, MonophonicMember


class LossHandler:
    """Calculates loss for a Song based on dissonance and auxiliary losses."""
    
    # Default loss factors - can be overridden via hyperparameters
    DEFAULT_LOSS_FACTORS = {
        # All members
        'self_concurrent': 1.0,
        'mate_concurrent': 0.16,
        'self_temporal': 0.4,
        'mate_temporal': 0.06,
        'extreme_range': 1.2,

        # Polyphonic members
        'quietness': 0.8,
        'muddyness': 0.3,
        'hand_stretch': 2.4,

        # Monophonic members
        'jump': 0.1,
        
    }
    
    def __init__(self, song: Song, max_hz: float = 11025):
        self.song = song
        self.max_hz = max_hz
        self.dissonance_matrices: Dict[Tuple[int, int], torch.Tensor] = {}
        
        # Precompute all dissonance matrices
        self._compute_dissonance_matrices()
    
    def _dissonance_function(self, f1: float, f2: float) -> float:
        """
        Calculate dissonance between two frequencies.
        Uses the formula: d = 65 * x * exp(-24 * x)
        where x = |f2 - f1| / min(f1, f2)
        Maximum dissonance at x ≈ 0.041667
        """
        if f1 <= 0 or f2 <= 0:
            return 0.0
        
        x = abs(f2 - f1) / min(f1, f2)
        d = 65 * x * np.exp(-24 * x)
        return d
    
    def _compute_dissonance_matrix(self, src_member: Member, dst_member: Member) -> torch.Tensor:
        """
        Compute dissonance matrix between two members.
        D[i, j] = dissonance caused by src's key i to dst's key j
        Includes harmonic interactions.
        """
        src_keys = src_member.num_keys
        dst_keys = dst_member.num_keys
        D = torch.zeros((src_keys, dst_keys))
        
        # Get mean amplitudes for each harmonic of each member
        src_mean_amps = src_member.instrument.mean_amplitudes(
            src_member.note_duration(), 
            self.song.sample_rate
        )
        dst_mean_amps = dst_member.instrument.mean_amplitudes(
            dst_member.note_duration(),
            self.song.sample_rate
        )
        
        for src_key in range(src_keys):
            src_base_freq = src_member.tuning_system.key_to_freq(
                src_member.instrument_range[0] + src_key
            )
            if src_base_freq is None:
                continue
                
            for dst_key in range(dst_keys):
                dst_base_freq = dst_member.tuning_system.key_to_freq(
                    dst_member.instrument_range[0] + dst_key
                )
                if dst_base_freq is None:
                    continue
                
                total_d = 0.0
                
                # Calculate dissonance between all harmonic pairs
                for h1_freq_mult, h1_amp in src_mean_amps:
                    for h2_freq_mult, h2_amp in dst_mean_amps:
                        f1 = src_base_freq * h1_freq_mult
                        f2 = dst_base_freq * h2_freq_mult
                        
                        # Skip if above Nyquist frequency
                        if f1 >= self.max_hz or f2 >= self.max_hz:
                            continue
                        
                        # Skip same note with same harmonic (no self-dissonance)
                        if src_member == dst_member and src_key == dst_key and h1_freq_mult == h2_freq_mult:
                            continue
                        
                        # Calculate dissonance for this pair of partials
                        d = self._dissonance_function(f1, f2)
                        
                        # Weight by harmonic strengths
                        total_d += d * h1_amp * h2_amp
                
                D[src_key, dst_key] = total_d
        
        return D
    
    def _compute_dissonance_matrices(self):
        """Precompute all pairwise dissonance matrices between members."""
        for src_idx, src_member in enumerate(self.song.members):
            for dst_idx, dst_member in enumerate(self.song.members):
                D = self._compute_dissonance_matrix(src_member, dst_member)
                self.dissonance_matrices[(src_idx, dst_idx)] = D
    
    def sample_like(
        self, 
        dst_weights: torch.Tensor, 
        src: Member, 
        dst: Member,
        note_shift: int = 0
    ) -> torch.Tensor:
        """
        Resample dst_weights to match src's time scale.
        
        This performs energy-preserving horizontal scaling and cropping:
        - Stretch dst weights by factor of dst.ticks_per_note to song time
        - Crop by note_shift * src.ticks_per_note
        - Squash by factor of src.ticks_per_note to src note times
        
        Args:
            dst_weights: (dst_keys, dst_notes) tensor
            src: source member (determines output time scale)
            dst: destination member (determines input time scale)
            note_shift: -1 for past, 0 for concurrent, +1 for future
            
        Returns:
            (dst_keys, src_notes - abs(note_shift)) tensor
        """
        dst_keys, dst_notes = dst_weights.shape
        
        # Stretch: expand each note to ticks_per_note samples
        # (dst_keys, dst_notes) -> (dst_keys, dst_notes * dst.ticks_per_note)
        stretch_factor = dst.ticks_per_note
        stretched = dst_weights.repeat_interleave(stretch_factor, dim=1)
        
        # Calculate crop amounts
        total_song_ticks = self.song.total_ticks()
        crop_amount = abs(note_shift) * src.ticks_per_note
        
        if note_shift < 0:
            # Crop from the left (past notes)
            if crop_amount > 0:
                stretched = stretched[:, crop_amount:]
        elif note_shift > 0:
            # Crop from the right (future notes)
            if crop_amount > 0:
                stretched = stretched[:, :-crop_amount]
        
        # Squash: average every ticks_per_note samples
        # (dst_keys, cropped_ticks) -> (dst_keys, src_notes)
        squashed_length = stretched.shape[1] // src.ticks_per_note
        squashed = torch.zeros((dst_keys, squashed_length))
        
        for i in range(squashed_length):
            start = i * src.ticks_per_note
            end = start + src.ticks_per_note
            # Average over the time period (preserves energy in terms of sum)
            squashed[:, i] = stretched[:, start:end].sum(dim=1)
        
        return squashed
    
    def _get_activation(self, member: Member) -> torch.Tensor:
        """Get the activated weights for a member."""
        if isinstance(member, PolyphonicMember):
            # ReLU activation
            return torch.relu(member.weights)
        elif isinstance(member, MonophonicMember):
            # Gumbel-softmax (straight-through estimator)
            # Forward: argmax, Backward: softmax with temperature
            logits = member.weights  # (keys, notes)
            
            # During forward pass for loss computation, use soft probabilities
            # Temperature can be adjusted - lower = more discrete
            temperature = member.hp.get('gumbel_temperature', 0.5)
            
            # Apply softmax across keys dimension
            probs = F.softmax(logits / temperature, dim=0)
            
            # Straight-through estimator: forward uses hard, backward uses soft
            hard = torch.zeros_like(logits)
            max_indices = torch.argmax(logits, dim=0)
            hard.scatter_(0, max_indices.unsqueeze(0), 1.0)
            
            # Use hard for forward, soft for backward
            activation = hard + (probs - probs.detach())
            return activation
        else:
            raise ValueError(f"Unknown member type: {type(member)}")
    
    def calculate_loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total loss and return loss components.
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary of individual loss components for logging
        """
        total_loss = torch.tensor(0.0)
        loss_dict = {}
        
        for src_idx, src in enumerate(self.song.members):
            src_activation = self._get_activation(src)
            
            # Get loss factors from member hyperparameters or defaults
            factors = {k: src.hp.get(f'loss_{k}', v) 
                      for k, v in self.DEFAULT_LOSS_FACTORS.items()}
            
            # === Self-concurrent loss (polyphonic only) ===
            if isinstance(src, PolyphonicMember):
                D_self = self.dissonance_matrices[(src_idx, src_idx)]
                # torch.sum(src.T @ D @ src)
                # Trace is dissonance at each note
                # But since we're separating concurrent and temporal (off-trace), do a for loop over notes
                self_concurrent = 0.0
                for note in range(src.num_notes):
                    note_weights = src_activation[:, note]
                    self_concurrent += torch.sum(note_weights @ D_self @ note_weights)
                    # Note that the first note is not the recipient of any temporal dissonance
                    # To make things fair, double its concurrent dissonance
                    # Awful workaround
                    if note == 0:
                        self_concurrent *= 2.0
                
                loss_component = factors['self_concurrent'] * self_concurrent
                total_loss = total_loss + loss_component
                loss_dict[f'{src.name}_self_concurrent'] = self_concurrent.item()
            
            # === Mate-concurrent loss ===
            mate_concurrent = 0.0
            for dst_idx, dst in enumerate(self.song.members):
                if src_idx == dst_idx:
                    continue
                
                D_mate = self.dissonance_matrices[(src_idx, dst_idx)]
                dst_activation = self._get_activation(dst)
                
                # Resample dst to src's time scale
                dst_sampled = self.sample_like(dst_activation, src, dst, note_shift=0)
                
                # Calculate dissonance for overlapping notes
                min_notes = min(src_activation.shape[1], dst_sampled.shape[1])
                for note in range(min_notes):
                    src_weights = src_activation[:, note]
                    dst_weights = dst_sampled[:, note]
                    mate_concurrent += torch.sum(src_weights @ D_mate @ dst_weights)
            
            loss_component = factors['mate_concurrent'] * mate_concurrent
            total_loss = total_loss + loss_component
            loss_dict[f'{src.name}_mate_concurrent'] = mate_concurrent
            
            # === Self-temporal loss ===
            self_temporal = 0.0
            if src.num_notes > 1:
                D_self = self.dissonance_matrices[(src_idx, src_idx)]
                # Dissonance between consecutive notes
                for note in range(1, src.num_notes):
                    curr_weights = src_activation[:, note]
                    prev_weights = src_activation[:, note - 1]
                    self_temporal += torch.sum(curr_weights @ D_self @ prev_weights)
                
                loss_component = factors['self_temporal'] * self_temporal
                total_loss = total_loss + loss_component
                loss_dict[f'{src.name}_self_temporal'] = self_temporal
            
            # === Mate-temporal loss ===
            mate_temporal = 0.0
            for dst_idx, dst in enumerate(self.song.members):
                D_mate = self.dissonance_matrices[(src_idx, dst_idx)]
                dst_activation = self._get_activation(dst)
                
                # Resample dst with a shift of -1 (past note)
                dst_sampled = self.sample_like(dst_activation, src, dst, note_shift=-1)
                
                # Calculate dissonance between current src and previous dst
                min_notes = min(src_activation.shape[1] - 1, dst_sampled.shape[1])
                if min_notes > 0:
                    for note in range(min_notes):
                        curr_src = src_activation[:, note + 1]
                        prev_dst = dst_sampled[:, note]
                        mate_temporal += torch.sum(curr_src @ D_mate @ prev_dst)
            
            loss_component = factors['mate_temporal'] * mate_temporal
            total_loss = total_loss + loss_component
            loss_dict[f'{src.name}_mate_temporal'] = mate_temporal
            
            # === Auxiliary losses ===
            
            # Extreme range loss (prefer middle of range)
            key_indices = torch.arange(src.num_keys, dtype=torch.float32)
            middle_key = (src.num_keys - 1) / 2
            distances_from_middle = torch.abs(key_indices - middle_key) / middle_key
            
            # Weighted average of distances, weighted by activation
            total_extreme_range = 0.0
            for note in range(src.num_notes):
                extreme_range = torch.sum(distances_from_middle * src_activation[:, note])
                loss_component = factors['extreme_range'] * extreme_range
                total_loss = total_loss + loss_component
                total_extreme_range += extreme_range.item()
            
            if src.num_notes > 0:
                loss_dict[f'{src.name}_extreme_range'] = total_extreme_range / src.num_notes
            else:
                loss_dict[f'{src.name}_extreme_range'] = 0.0
            
            if isinstance(src, PolyphonicMember):
                # Quietness loss (inverse L2, encourage some activity)
                # use square so stronger notes contribute more gain
                # But clamp it because too many notes near 1 causes clipping
                quietness = 2.0 * -torch.sum(torch.square(torch.clamp(src_activation, min=0.0, max=0.5)))
                loss_component = factors['quietness'] * quietness
                total_loss = total_loss + loss_component
                loss_dict[f'{src.name}_quietness'] = quietness.item()

                # Muddyness loss (L1 regularization to encourage sparsity)
                muddyness = torch.sum(torch.abs(src_activation))
                loss_component = factors['muddyness'] * muddyness
                total_loss = total_loss + loss_component
                loss_dict[f'{src.name}_muddyness'] = muddyness.item()
                
                # Hand stretch loss (encourage notes close together)
                total_hand_stretch = 0.0
                stretch_count = 0
                if src.num_notes > 0:
                    for note in range(src.num_notes):
                        active_keys = torch.where(src_activation[:, note] > 0)[0]
                        if len(active_keys) > 1:
                            key_spread = torch.max(active_keys).float() - torch.min(active_keys).float()
                            hand_stretch = key_spread / src.num_keys
                            loss_component = factors['hand_stretch'] * hand_stretch
                            total_loss = total_loss + loss_component
                            total_hand_stretch += hand_stretch.item()
                            stretch_count += 1
                
                if stretch_count > 0:
                    loss_dict[f'{src.name}_hand_stretch'] = total_hand_stretch / stretch_count
                else:
                    loss_dict[f'{src.name}_hand_stretch'] = 0.0
            
            elif isinstance(src, MonophonicMember):
                # Jump loss (penalize large jumps between consecutive notes)
                if src.num_notes > 1:
                    jump_loss = 0.0
                    for note in range(1, src.num_notes):
                        # Get the active key for each note
                        prev_key = torch.argmax(src_activation[:, note - 1]).float()
                        curr_key = torch.argmax(src_activation[:, note]).float()
                        jump = torch.abs(curr_key - prev_key) / src.num_keys
                        jump_loss += jump
                    
                    loss_component = factors['jump'] * jump_loss
                    total_loss = total_loss + loss_component
                    loss_dict[f'{src.name}_jump'] = jump_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
