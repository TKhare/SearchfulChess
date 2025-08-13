#!/usr/bin/env python3
"""
Test script to verify puzzle setup works correctly.
"""

import sys
import os

# Add paths
engine_src = os.path.join(os.path.dirname(__file__), 'src')
searchless_src = os.path.join(os.path.dirname(__file__), '../searchless_chess/src')
sys.path.extend([engine_src, searchless_src])

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import chess
        print("✓ chess library imported")
    except ImportError as e:
        print(f"✗ Failed to import chess: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        from src.hybrid_engine import HybridEngine
        print("✓ HybridEngine imported")
    except ImportError as e:
        print(f"✗ Failed to import HybridEngine: {e}")
        return False
    
    try:
        from src.puzzle_adapter import HybridEngineAdapter, register_hybrid_engines
        print("✓ Puzzle adapter imported")
    except ImportError as e:
        print(f"✗ Failed to import puzzle adapter: {e}")
        return False
    
    try:
        from engines import constants as engine_constants
        print("✓ DeepMind engine constants imported")
    except ImportError as e:
        print(f"✗ Failed to import engine constants: {e}")
        return False
    
    try:
        from puzzles import evaluate_puzzle_from_board
        print("✓ DeepMind puzzle evaluation imported")
    except ImportError as e:
        print(f"✗ Failed to import puzzle evaluation: {e}")
        return False
    
    return True


def test_hybrid_engine():
    """Test that hybrid engine can be created and used."""
    print("\nTesting hybrid engine creation...")
    
    try:
        from src.hybrid_engine import HybridEngine
        
        # Try to create a simple hybrid engine
        print("Creating HybridEngine with 9M_state_value...")
        engine = HybridEngine(model_name='9M_state_value', max_depth=2, time_limit=5.0)
        print("✓ HybridEngine created successfully")
        
        # Test on a simple position
        import chess
        board = chess.Board()
        print("Testing move generation...")
        
        move, eval_score, stats = engine.search(board, depth=2)
        print(f"✓ Engine found move: {move} (eval: {eval_score:.3f})")
        print(f"  Search stats: {stats.nodes_searched} nodes, {stats.evaluations_made} evals")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test hybrid engine: {e}")
        return False


def test_puzzle_adapter():
    """Test that puzzle adapter works."""
    print("\nTesting puzzle adapter...")
    
    try:
        from src.puzzle_adapter import HybridEngineAdapter, register_hybrid_engines
        
        # Register engines
        print("Registering hybrid engines...")
        success = register_hybrid_engines()
        if not success:
            print("✗ Failed to register engines")
            return False
        print("✓ Engines registered")
        
        # Create adapter
        print("Creating HybridEngineAdapter...")
        adapter = HybridEngineAdapter(model_name='9M_state_value', max_depth=2, time_limit=3.0)
        print("✓ Adapter created")
        
        # Test play method
        import chess
        board = chess.Board()
        print("Testing adapter.play() method...")
        move = adapter.play(board)
        print(f"✓ Adapter found move: {move}")
        
        # Test analyse method
        print("Testing adapter.analyse() method...")
        analysis = adapter.analyse(board)
        print(f"✓ Analysis complete: {analysis['best_move']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test puzzle adapter: {e}")
        return False


def test_engine_registration():
    """Test that engines are properly registered."""
    print("\nTesting engine registration...")
    
    try:
        from src.puzzle_adapter import register_hybrid_engines
        from engines import constants as engine_constants
        
        # Register engines
        register_hybrid_engines()
        
        # Check that hybrid engines are available
        hybrid_engines = [name for name in engine_constants.ENGINE_BUILDERS.keys() 
                         if name.startswith('hybrid_')]
        
        print(f"Found {len(hybrid_engines)} hybrid engines:")
        for engine_name in hybrid_engines:
            print(f"  - {engine_name}")
        
        if len(hybrid_engines) == 0:
            print("✗ No hybrid engines found")
            return False
        
        # Test creating one of the engines
        test_engine_name = hybrid_engines[0]
        print(f"Testing creation of {test_engine_name}...")
        engine = engine_constants.ENGINE_BUILDERS[test_engine_name]()
        print(f"✓ Successfully created {test_engine_name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test engine registration: {e}")
        return False


def check_puzzle_data():
    """Check if puzzle data is available."""
    print("\nChecking puzzle data availability...")
    
    puzzles_path = os.path.join(os.path.dirname(__file__), '../searchless_chess/data/puzzles.csv')
    
    if os.path.exists(puzzles_path):
        print(f"✓ Puzzle data found at: {puzzles_path}")
        
        # Check file size
        file_size = os.path.getsize(puzzles_path)
        print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
        
        # Try to read first few lines
        try:
            import pandas as pd
            sample = pd.read_csv(puzzles_path, nrows=3)
            print(f"  Sample data shape: {sample.shape}")
            print(f"  Columns: {list(sample.columns)}")
            return True
        except Exception as e:
            print(f"  ✗ Error reading puzzle data: {e}")
            return False
    else:
        print(f"✗ Puzzle data not found at: {puzzles_path}")
        print("  To download:")
        print("  cd ../searchless_chess/data")
        print("  wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HYBRID ENGINE PUZZLE SETUP TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Hybrid Engine", test_hybrid_engine),
        ("Puzzle Adapter", test_puzzle_adapter),
        ("Engine Registration", test_engine_registration),
        ("Puzzle Data", check_puzzle_data),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name:=^60}")
        success = test_func()
        results.append((test_name, success))
        print(f"{'PASS' if success else 'FAIL':^60}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<30} {status:>10}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to run puzzle evaluation.")
        print("\nTo run puzzles:")
        print("  python puzzle_runner.py --engine hybrid_9M_state_value --num_puzzles 10")
        print("  python puzzle_runner.py --compare 9M hybrid_9M_state_value --num_puzzles 10")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before running puzzles.")


if __name__ == '__main__':
    main() 