#!/usr/bin/env python3
"""
Demo script for the Hybrid Chess Engine.

This script demonstrates the hybrid engine's capabilities by:
1. Loading the transformer model
2. Analyzing famous chess positions
3. Comparing performance with different search depths
4. Showing the search tree evaluation process
"""

import sys
import os
import chess
import time

# Add the engine source to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_engine import HybridEngine


def analyze_famous_positions():
    """Analyze some famous chess positions to demonstrate engine capability."""
    
    famous_positions = [
        {
            'name': 'Starting Position',
            'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'description': 'The initial chess position'
        },
        {
            'name': 'Sicilian Defense',
            'fen': 'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
            'description': 'After 1.e4 c5 - popular opening'
        },
        {
            'name': 'Queen\'s Gambit',
            'fen': 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2',
            'description': 'After 1.d4 d5 2.c4 - famous opening'
        },
        {
            'name': 'Tactical Position',
            'fen': 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4',
            'description': 'A tactical middlegame position'
        },
        {
            'name': 'Endgame Position',
            'fen': '8/8/8/8/8/3k4/3P4/3K4 w - - 0 1',
            'description': 'King and pawn vs king endgame'
        }
    ]
    
    print("=== Hybrid Chess Engine Demo ===\n")
    print("Analyzing famous chess positions with transformer evaluation...\n")
    
    # Initialize engine with 9M state value model at moderate depth
    engine = HybridEngine(model_name='9M_state_value', max_depth=4, time_limit=30.0)
    
    for i, position in enumerate(famous_positions, 1):
        print(f"{i}. {position['name']}")
        print(f"   Description: {position['description']}")
        print(f"   FEN: {position['fen']}")
        
        board = chess.Board(position['fen'])
        
        # Display the position
        print(f"   Position:")
        print("   " + "\n   ".join(str(board).split('\n')))
        
        try:
            # Analyze the position
            print(f"\n   Analyzing with transformer evaluation...")
            start_time = time.time()
            
            analysis = engine.analyze_position(board, depth=3)
            
            analysis_time = time.time() - start_time
            
            print(f"   Best move: {analysis['best_move']}")
            print(f"   Evaluation: {analysis['evaluation']:.3f}")
            print(f"   Analysis time: {analysis_time:.2f}s")
            print(f"   Nodes searched: {analysis['search_stats'].nodes_searched}")
            print(f"   Neural evaluations: {analysis['search_stats'].evaluations_made}")
            print(f"   Alpha-beta cutoffs: {analysis['search_stats'].alpha_beta_cutoffs}")
            
        except Exception as e:
            print(f"   Error analyzing position: {e}")
        
        print("\n" + "="*60 + "\n")


def compare_search_depths():
    """Compare engine performance at different search depths."""
    
    print("=== Search Depth Comparison ===\n")
    
    # Use a tactical position for comparison
    test_fen = 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4'
    board = chess.Board(test_fen)
    
    print(f"Test position: {test_fen}")
    print("Position:")
    print("\n".join("  " + line for line in str(board).split('\n')))
    print()
    
    depths = [2, 3, 4, 5]
    
    for depth in depths:
        print(f"Depth {depth}:")
        
        # Create engine for this depth
        engine = HybridEngine(model_name='9M_state_value', max_depth=depth, time_limit=60.0)
        
        try:
            start_time = time.time()
            move, evaluation, stats = engine.search(board, depth=depth)
            analysis_time = time.time() - start_time
            
            print(f"  Best move: {move}")
            print(f"  Evaluation: {evaluation:.3f}")
            print(f"  Time: {analysis_time:.2f}s")
            print(f"  Nodes: {stats.nodes_searched}")
            print(f"  Evaluations: {stats.evaluations_made}")
            print(f"  Cutoffs: {stats.alpha_beta_cutoffs}")
            print(f"  Eval rate: {stats.evaluations_made/analysis_time:.1f} evals/sec")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        print()


def interactive_mode():
    """Interactive mode where user can input positions."""
    
    print("=== Interactive Mode ===\n")
    print("Enter FEN positions to analyze (or 'quit' to exit)")
    print("Press Enter for starting position\n")
    
    engine = HybridEngine(model_name='9M_state_value', max_depth=4, time_limit=30.0)
    
    while True:
        fen_input = input("FEN> ").strip()
        
        if fen_input.lower() in ['quit', 'exit', 'q']:
            break
        
        # Default to starting position
        if not fen_input:
            fen_input = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        
        try:
            board = chess.Board(fen_input)
            
            print(f"\nPosition: {fen_input}")
            print("\n".join("  " + line for line in str(board).split('\n')))
            
            analysis = engine.analyze_position(board)
            
            print(f"\nBest move: {analysis['best_move']}")
            print(f"Evaluation: {analysis['evaluation']:.3f}")
            print(f"Search stats: {analysis['search_stats'].nodes_searched} nodes, "
                  f"{analysis['search_stats'].evaluations_made} evals")
            
        except ValueError as e:
            print(f"Invalid FEN: {e}")
        except Exception as e:
            print(f"Analysis error: {e}")
        
        print()


def main():
    """Main demo function."""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'positions':
            analyze_famous_positions()
        elif mode == 'depths':
            compare_search_depths()
        elif mode == 'interactive':
            interactive_mode()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: positions, depths, interactive")
    else:
        # Default: run famous positions analysis
        analyze_famous_positions()


if __name__ == '__main__':
    main() 