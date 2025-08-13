#!/usr/bin/env python3
"""
Puzzle evaluation runner for the Hybrid Chess Engine.

This script runs DeepMind's puzzle evaluation system with our hybrid engine,
allowing direct comparison with their neural engines.
"""

import sys
import os
import argparse
import time
from typing import Sequence

# Add paths
engine_src = os.path.join(os.path.dirname(__file__), 'src')
searchless_src = os.path.join(os.path.dirname(__file__), '../searchless_chess/src')
sys.path.extend([engine_src, searchless_src])

# Import required modules
import chess
import chess.pgn
import pandas as pd
import io

# Import DeepMind's puzzle evaluation code
from puzzles import evaluate_puzzle_from_pandas_row, evaluate_puzzle_from_board
from engines import constants as engine_constants

# Import our hybrid engine components
from src.puzzle_adapter import HybridEngineAdapter, register_hybrid_engines, get_available_hybrid_engines


def download_puzzles_if_needed():
    """Download puzzle data if not available."""
    puzzles_path = os.path.join(os.path.dirname(__file__), '../searchless_chess/data/puzzles.csv')
    
    if not os.path.exists(puzzles_path):
        print("Puzzle data not found. You need to download it first.")
        print("Run this command:")
        print("cd ../searchless_chess/data && wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv")
        return None
    
    return puzzles_path


def run_puzzle_evaluation(
    engine_name: str,
    num_puzzles: int = 10,
    verbose: bool = True
):
    """
    Run puzzle evaluation with the specified engine.
    
    Args:
        engine_name: Name of the engine to use
        num_puzzles: Number of puzzles to evaluate
        verbose: Whether to print detailed results
        
    Returns:
        Dictionary with evaluation results
    """
    # Get puzzle data
    puzzles_path = download_puzzles_if_needed()
    if puzzles_path is None:
        return None
    
    print(f"Loading {num_puzzles} puzzles from {puzzles_path}")
    puzzles = pd.read_csv(puzzles_path, nrows=num_puzzles)
    
    # Register hybrid engines
    register_hybrid_engines()
    
    # Get the engine
    if engine_name not in engine_constants.ENGINE_BUILDERS:
        print(f"Error: Engine '{engine_name}' not found!")
        print("Available engines:")
        for name in sorted(engine_constants.ENGINE_BUILDERS.keys()):
            print(f"  - {name}")
        return None
    
    print(f"Creating engine: {engine_name}")
    engine = engine_constants.ENGINE_BUILDERS[engine_name]()
    
    # Track results
    results = []
    correct_count = 0
    total_time = 0
    
    print(f"\nEvaluating {len(puzzles)} puzzles with {engine_name}...")
    print("=" * 60)
    
    for puzzle_id, puzzle in puzzles.iterrows():
        start_time = time.time()
        
        try:
            correct = evaluate_puzzle_from_pandas_row(
                puzzle=puzzle,
                engine=engine,
            )
            
            solve_time = time.time() - start_time
            total_time += solve_time
            
            if correct:
                correct_count += 1
            
            result = {
                'puzzle_id': puzzle_id,
                'correct': correct,
                'rating': puzzle['Rating'],
                'solve_time': solve_time,
                'themes': puzzle.get('Themes', ''),
            }
            
            results.append(result)
            
            if verbose:
                status = "✓" if correct else "✗"
                print(f"Puzzle {puzzle_id:3d}: {status} (Rating: {puzzle['Rating']:4d}, Time: {solve_time:.2f}s)")
                
        except Exception as e:
            print(f"Error on puzzle {puzzle_id}: {e}")
            result = {
                'puzzle_id': puzzle_id,
                'correct': False,
                'rating': puzzle['Rating'],
                'solve_time': 0.0,
                'error': str(e)
            }
            results.append(result)
    
    # Calculate summary statistics
    accuracy = correct_count / len(puzzles) * 100
    avg_time = total_time / len(puzzles)
    avg_rating = sum(r['rating'] for r in results) / len(results)
    
    print("=" * 60)
    print(f"RESULTS SUMMARY:")
    print(f"Engine: {engine_name}")
    print(f"Puzzles solved: {correct_count}/{len(puzzles)} ({accuracy:.1f}%)")
    print(f"Average time per puzzle: {avg_time:.2f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average puzzle rating: {avg_rating:.0f}")
    
    return {
        'engine_name': engine_name,
        'correct_count': correct_count,
        'total_puzzles': len(puzzles),
        'accuracy': accuracy,
        'average_time': avg_time,
        'total_time': total_time,
        'average_rating': avg_rating,
        'detailed_results': results
    }


def compare_engines(
    engine_names: Sequence[str],
    num_puzzles: int = 10
):
    """
    Compare multiple engines on the same puzzle set.
    
    Args:
        engine_names: List of engine names to compare
        num_puzzles: Number of puzzles to evaluate
    """
    print(f"Comparing {len(engine_names)} engines on {num_puzzles} puzzles...")
    print("Engines:", ", ".join(engine_names))
    print()
    
    all_results = []
    
    for engine_name in engine_names:
        print(f"\nTesting {engine_name}...")
        print("-" * 40)
        
        result = run_puzzle_evaluation(
            engine_name=engine_name,
            num_puzzles=num_puzzles,
            verbose=False
        )
        
        if result:
            all_results.append(result)
    
    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Engine':<25} {'Solved':<10} {'Accuracy':<10} {'Avg Time':<10} {'Total Time':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['engine_name']:<25} "
              f"{result['correct_count']}/{result['total_puzzles']:<8} "
              f"{result['accuracy']:.1f}%{'':6} "
              f"{result['average_time']:.2f}s{'':5} "
              f"{result['total_time']:.1f}s")
    
    return all_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Hybrid Engine Puzzle Evaluation')
    
    parser.add_argument('--engine', type=str, default='hybrid_9M_state_value',
                        help='Engine to evaluate (default: hybrid_9M_state_value)')
    parser.add_argument('--num_puzzles', type=int, default=10,
                        help='Number of puzzles to evaluate (default: 10)')
    parser.add_argument('--compare', nargs='+',
                        help='Compare multiple engines (e.g., --compare 9M hybrid_9M)')
    parser.add_argument('--list_engines', action='store_true',
                        help='List all available engines')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed results')
    
    args = parser.parse_args()
    
    # Register hybrid engines first
    register_hybrid_engines()
    
    if args.list_engines:
        print("Available engines:")
        all_engines = sorted(engine_constants.ENGINE_BUILDERS.keys())
        
        # Separate hybrid engines
        hybrid_engines = [e for e in all_engines if e.startswith('hybrid_')]
        other_engines = [e for e in all_engines if not e.startswith('hybrid_')]
        
        print("\nHybrid Engines:")
        for engine in hybrid_engines:
            print(f"  - {engine}")
            
        print("\nOther Engines:")
        for engine in other_engines:
            print(f"  - {engine}")
        
        return
    
    if args.compare:
        # Compare multiple engines
        compare_engines(args.compare, args.num_puzzles)
    else:
        # Single engine evaluation
        run_puzzle_evaluation(
            engine_name=args.engine,
            num_puzzles=args.num_puzzles,
            verbose=args.verbose
        )


if __name__ == '__main__':
    main() 