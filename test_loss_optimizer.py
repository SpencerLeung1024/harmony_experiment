"""
Verification tests for losses, constraints, and optimizer modules.

This file tests:
1. Loss calculation with multiple members
2. Constraint application (fixed notes influence loss)
3. Full optimization loop
4. Cross-member dissonance calculation
"""

import torch
import numpy as np

from harmony import (
    # Band members
    PolyphonicMember,
    MonophonicMember,
    DrumMember,
    # Constraints
    UserConstraint,
    ConstraintSet,
    # Loss and optimization
    LossFunction,
    HarmonyOptimizer,
    # Tuning
    TwelveTET,
    PythagoreanTuning,
)


def test_loss_function_single_member():
    """Test loss function with a single member."""
    print("\n" + "=" * 60)
    print("TEST 1: Loss Function - Single Member")
    print("=" * 60)
    
    # Create a piano member
    piano = PolyphonicMember.piano(num_beats=4)
    
    # Create loss function
    loss_fn = LossFunction([piano])
    loss_fn.precompute_dissonance_matrices()
    
    # Calculate loss
    total_loss, breakdown = loss_fn.calculate()
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Breakdown: {breakdown}")
    
    # Verify all loss components exist
    assert 'within' in breakdown
    assert 'temporal' in breakdown
    assert 'density' in breakdown
    assert 'sparsity' in breakdown
    assert 'total' in breakdown
    
    # Verify loss is finite
    assert torch.isfinite(total_loss)
    
    print("✓ Single member loss calculation works")
    return True


def test_loss_function_multiple_members():
    """Test loss function with multiple band members."""
    print("\n" + "=" * 60)
    print("TEST 2: Loss Function - Multiple Members")
    print("=" * 60)
    
    # Create band members
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    bass = MonophonicMember.bass(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    
    members = [piano, guitar, bass, drums]
    
    # Create loss function
    loss_fn = LossFunction(members)
    loss_fn.precompute_dissonance_matrices()
    
    # Verify matrices computed
    assert len(loss_fn.dissonance_matrices) == 3  # piano, guitar, bass (no drums)
    assert len(loss_fn.cross_dissonance_matrices) == 3  # 3 choose 2 pairs
    
    # Calculate loss
    total_loss, breakdown = loss_fn.calculate()
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Within-member matrices: {len(loss_fn.dissonance_matrices)}")
    print(f"Cross-member matrices: {len(loss_fn.cross_dissonance_matrices)}")
    print(f"Breakdown: {breakdown}")
    
    assert 'cross' in breakdown
    assert torch.isfinite(total_loss)
    
    print("✓ Multiple member loss calculation works")
    return True


def test_cross_member_dissonance():
    """Test cross-member dissonance calculation."""
    print("\n" + "=" * 60)
    print("TEST 3: Cross-Member Dissonance")
    print("=" * 60)
    
    # Create two members with different ranges
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    
    loss_fn = LossFunction([piano, guitar])
    loss_fn.precompute_dissonance_matrices()
    
    # Get cross-dissonance matrix
    D_cross = loss_fn.cross_dissonance_matrices[('piano', 'guitar')]
    
    print(f"Piano keys: {piano.num_keys}")
    print(f"Guitar keys: {guitar.num_keys}")
    print(f"Cross-dissonance matrix shape: {D_cross.shape}")
    
    # Verify shape
    assert D_cross.shape == (piano.num_keys, guitar.num_keys)
    
    # Verify non-negative
    assert (D_cross >= 0).all()
    
    # Calculate cross-member loss
    cross_loss = loss_fn.calculate_cross_member()
    print(f"Cross-member loss: {cross_loss.item():.4f}")
    
    assert torch.isfinite(cross_loss)
    
    print("✓ Cross-member dissonance calculation works")
    return True


def test_constraints():
    """Test constraint system."""
    print("\n" + "=" * 60)
    print("TEST 4: User Constraints")
    print("=" * 60)
    
    # Create constraint set
    constraints = ConstraintSet()
    
    # Add single note constraint
    constraints.add_note("piano", beat_index=0, key_index=60, strength=1.0)
    
    # Add chord constraint
    constraints.add_chord("piano", beat_index=1, key_indices=[64, 67, 71])
    
    # Add guitar constraint
    constraints.add_note("guitar", beat_index=0, key_index=40, strength=0.8)
    
    print(f"Total constraints: {len(constraints)}")
    
    # Create piano member
    piano = PolyphonicMember.piano(num_beats=4)
    
    # Apply constraints
    constraint_matrix = constraints.apply_to_member(piano)
    print(f"Constraint matrix shape: {constraint_matrix.shape}")
    print(f"Non-zero at beat 0: {torch.count_nonzero(constraint_matrix[:, 0])}")
    print(f"Non-zero at beat 1: {torch.count_nonzero(constraint_matrix[:, 1])}")
    
    # Verify constraints applied correctly
    assert constraint_matrix[60, 0] == 1.0
    assert constraint_matrix[64, 1] == 1.0
    assert constraint_matrix[67, 1] == 1.0
    assert constraint_matrix[71, 1] == 1.0
    
    # Test effective weights
    effective = constraints.get_effective_weights(piano)
    print(f"Effective weights shape: {effective.shape}")
    print(f"Key 60 at beat 0: {effective[60, 0].item():.4f}")
    
    # Verify constraints are detached
    assert not constraint_matrix.requires_grad
    
    print("✓ Constraint system works")
    return True


def test_constraint_gradient_flow():
    """Test that constraints don't receive gradients."""
    print("\n" + "=" * 60)
    print("TEST 5: Constraint Gradient Flow")
    print("=" * 60)
    
    # Create piano member
    piano = PolyphonicMember.piano(num_beats=4)
    
    # Create constraint
    constraints = ConstraintSet()
    constraints.add_note("piano", beat_index=0, key_index=60, strength=1.0)
    
    # Get effective weights
    effective = constraints.get_effective_weights(piano)
    
    # Calculate simple loss
    loss = effective.sum()
    loss.backward()
    
    # Check that piano weights have gradients
    assert piano.weights.grad is not None
    
    # Check that constraint matrix does not require grad
    constraint_matrix = constraints.apply_to_member(piano)
    assert not constraint_matrix.requires_grad
    
    print(f"Piano weights grad exists: {piano.weights.grad is not None}")
    print(f"Grad at [60, 0]: {piano.weights.grad[60, 0].item():.4f}")
    print(f"Grad at [0, 0]: {piano.weights.grad[0, 0].item():.4f}")
    
    print("✓ Constraints are correctly detached from gradient flow")
    return True


def test_optimizer_creation():
    """Test HarmonyOptimizer creation."""
    print("\n" + "=" * 60)
    print("TEST 6: Optimizer Creation")
    print("=" * 60)
    
    # Create band
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    drums = DrumMember.standard_rock(num_beats=4)
    
    members = [piano, guitar, drums]
    
    # Create optimizer
    optimizer = HarmonyOptimizer(members, lr=0.02)
    
    print(f"Members: {[m.name for m in members]}")
    print(f"Optimizable members: {[m.name for m in optimizer.optimizable_members]}")
    print(f"Learning rate: {optimizer.lr}")
    
    # Verify drums excluded from optimization
    assert len(optimizer.optimizable_members) == 2
    assert all(not isinstance(m, DrumMember) for m in optimizer.optimizable_members)
    
    print("✓ Optimizer creation works")
    return True


def test_optimizer_step():
    """Test single optimization step."""
    print("\n" + "=" * 60)
    print("TEST 7: Optimizer Step")
    print("=" * 60)
    
    # Create band
    piano = PolyphonicMember.piano(num_beats=4)
    guitar = MonophonicMember.guitar(num_beats=4)
    
    optimizer = HarmonyOptimizer([piano, guitar], lr=0.02)
    optimizer.precompute_dissonance()
    
    # Get initial loss
    initial_loss = optimizer.step()
    
    print(f"Initial loss: {initial_loss['total']:.4f}")
    print(f"Breakdown: {initial_loss}")
    print(f"Iteration: {optimizer.iteration}")
    
    # Verify loss components
    assert 'within' in initial_loss
    assert 'temporal' in initial_loss
    assert 'cross' in initial_loss
    assert 'density' in initial_loss
    assert 'sparsity' in initial_loss
    
    # Verify iteration counter
    assert optimizer.iteration == 1
    
    print("✓ Optimizer step works")
    return True


def test_optimization_loop():
    """Test full optimization loop."""
    print("\n" + "=" * 60)
    print("TEST 8: Optimization Loop")
    print("=" * 60)
    
    # Create smaller band for faster test (use direct constructor for smaller range)
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    guitar = MonophonicMember(
        name="guitar",
        instrument=Instrument.guitar(),
        tuning=tuning,
        num_keys=24,
        key_offset=40,
        num_beats=4
    )
    
    optimizer = HarmonyOptimizer([piano, guitar], lr=0.02)
    optimizer.precompute_dissonance()
    
    # Run optimization
    result = optimizer.optimize(steps=20, verbose=False)
    
    print(f"Steps: {result['steps']}")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Loss history length: {len(result['loss_history'])}")
    
    # Verify loss decreased (or at least stayed finite)
    assert result['final_loss'] < float('inf')
    assert len(result['loss_history']) == 20
    
    # Check that loss history contains all expected keys
    first_step = result['loss_history'][0]
    assert 'total' in first_step
    assert 'within' in first_step
    
    print("✓ Optimization loop works")
    return True


def test_optimization_with_constraints():
    """Test optimization with user constraints."""
    print("\n" + "=" * 60)
    print("TEST 9: Optimization With Constraints")
    print("=" * 60)
    
    # Create band (use direct constructor for smaller range)
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    guitar = MonophonicMember(
        name="guitar",
        instrument=Instrument.guitar(),
        tuning=tuning,
        num_keys=24,
        key_offset=40,
        num_beats=4
    )
    
    # Create constraints - fix a C major chord on beat 0
    constraints = ConstraintSet()
    constraints.add_chord("piano", beat_index=0, key_indices=[0, 4, 7])  # C, E, G relative to offset
    
    # Create optimizer with constraints
    optimizer = HarmonyOptimizer(
        [piano, guitar],
        constraints=constraints,
        lr=0.02
    )
    optimizer.precompute_dissonance()
    
    # Run optimization
    result = optimizer.optimize(steps=20, verbose=False)
    
    # Check that constraint is still present in final result
    effective_weights = optimizer.constraints.get_effective_weights(piano)
    
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Constraint at beat 0, key 0: {effective_weights[0, 0].item():.4f}")
    print(f"Constraint at beat 0, key 4: {effective_weights[4, 0].item():.4f}")
    print(f"Constraint at beat 0, key 7: {effective_weights[7, 0].item():.4f}")
    
    # Verify constraints are present
    assert effective_weights[0, 0] >= 1.0
    assert effective_weights[4, 0] >= 1.0
    assert effective_weights[7, 0] >= 1.0
    
    print("✓ Optimization with constraints works")
    return True


def test_different_tuning_systems():
    """Test optimization with different tuning systems."""
    print("\n" + "=" * 60)
    print("TEST 10: Different Tuning Systems")
    print("=" * 60)
    
    # Create members with different tunings
    from harmony import Instrument
    twelve_tet = TwelveTET()
    pythagorean = PythagoreanTuning()
    
    piano_12tet = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=twelve_tet,
        num_keys=12,
        key_offset=48,
        num_beats=4
    )
    guitar_pyth = MonophonicMember(
        name="guitar",
        instrument=Instrument.guitar(),
        tuning=pythagorean,
        num_keys=12,
        key_offset=40,
        num_beats=4
    )
    
    optimizer = HarmonyOptimizer([piano_12tet, guitar_pyth], lr=0.02)
    optimizer.precompute_dissonance()
    
    # Run short optimization
    result = optimizer.optimize(steps=10, verbose=False)
    
    print(f"12-TET piano + Pythagorean guitar")
    print(f"Final loss: {result['final_loss']:.4f}")
    
    assert result['final_loss'] < float('inf')
    
    print("✓ Different tuning systems work")
    return True


def test_loss_weights():
    """Test custom loss weights."""
    print("\n" + "=" * 60)
    print("TEST 11: Custom Loss Weights")
    print("=" * 60)
    
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    
    # Custom weights emphasizing within-member dissonance
    custom_weights = {
        'within': 2.0,
        'temporal': 0.5,
        'density': 5.0
    }
    
    optimizer = HarmonyOptimizer(
        [piano],
        loss_weights=custom_weights,
        lr=0.02
    )
    optimizer.precompute_dissonance()
    
    result = optimizer.optimize(steps=10, verbose=False)
    
    print(f"Custom weights: {custom_weights}")
    print(f"Final loss: {result['final_loss']:.4f}")
    
    assert result['final_loss'] < float('inf')
    
    print("✓ Custom loss weights work")
    return True


def test_chord_analysis():
    """Test chord analysis functionality."""
    print("\n" + "=" * 60)
    print("TEST 12: Chord Analysis")
    print("=" * 60)
    
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    guitar = MonophonicMember(
        name="guitar",
        instrument=Instrument.guitar(),
        tuning=tuning,
        num_keys=24,
        key_offset=40,
        num_beats=4
    )
    
    optimizer = HarmonyOptimizer([piano, guitar], lr=0.02)
    optimizer.precompute_dissonance()
    optimizer.optimize(steps=10, verbose=False)
    
    # Get chord analysis
    analysis = optimizer.get_chord_analysis(threshold=0.2)
    
    print(f"Analyzed {len(analysis['members'])} members")
    for member_name, member_analysis in analysis['members'].items():
        print(f"  {member_name}: {len(member_analysis['chords'])} beats")
    
    assert 'members' in analysis
    assert 'global_progression' in analysis
    
    print("✓ Chord analysis works")
    return True


def test_active_notes():
    """Test getting active notes."""
    print("\n" + "=" * 60)
    print("TEST 13: Active Notes")
    print("=" * 60)
    
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    drums = DrumMember.standard_rock(num_beats=4)
    
    optimizer = HarmonyOptimizer([piano, drums], lr=0.02)
    optimizer.precompute_dissonance()
    optimizer.optimize(steps=10, verbose=False)
    
    # Get active notes
    active = optimizer.get_active_notes(threshold=0.2)
    
    print(f"Active notes for {len(active)} members:")
    for name, notes in active.items():
        total = sum(len(n) for n in notes)
        print(f"  {name}: {total} notes across {len(notes)} beats")
    
    assert 'piano' in active
    assert 'drums' in active
    
    print("✓ Getting active notes works")
    return True


def test_optimizer_reset():
    """Test optimizer reset functionality."""
    print("\n" + "=" * 60)
    print("TEST 14: Optimizer Reset")
    print("=" * 60)
    
    from harmony import Instrument, TwelveTET
    tuning = TwelveTET()
    piano = PolyphonicMember(
        name="piano",
        instrument=Instrument.piano(),
        tuning=tuning,
        num_keys=24,
        key_offset=48,
        num_beats=4
    )
    
    optimizer = HarmonyOptimizer([piano], lr=0.02)
    optimizer.precompute_dissonance()
    
    # Run some steps
    optimizer.optimize(steps=10, verbose=False)
    
    print(f"Before reset - Iteration: {optimizer.iteration}")
    print(f"Before reset - Loss history: {len(optimizer.loss_history)}")
    
    # Reset
    optimizer.reset()
    
    print(f"After reset - Iteration: {optimizer.iteration}")
    print(f"After reset - Loss history: {len(optimizer.loss_history)}")
    
    assert optimizer.iteration == 0
    assert len(optimizer.loss_history) == 0
    
    print("✓ Optimizer reset works")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        test_loss_function_single_member,
        test_loss_function_multiple_members,
        test_cross_member_dissonance,
        test_constraints,
        test_constraint_gradient_flow,
        test_optimizer_creation,
        test_optimizer_step,
        test_optimization_loop,
        test_optimization_with_constraints,
        test_different_tuning_systems,
        test_loss_weights,
        test_chord_analysis,
        test_active_notes,
        test_optimizer_reset,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
