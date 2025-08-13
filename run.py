#!/usr/bin/env python3
"""
Main entry point for the Hybrid Chess Engine.

This script provides a command-line interface to run the hybrid engine
with various options and modes.
"""

import sys
import os
import argparse
import chess
import time

# Add the engine source to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from hybrid_engine import HybridEngine


def play_game():
    """Play a game against the hybrid engine."""
    print("=== Play Against Hybrid Engine ===\n")
    print("Enter moves in standard algebraic notation (e.g., 'e4', 'Nf3')")
    print("Type 'quit' to exit, 'help' for help\n")
    
    board = chess.Board()
    engine = HybridEngine(model_name='9M', max_depth=4, time_limit=15.0)
    
    while not board.is_game_over():
        print(f"Position: {board.fen()}")
        print(board)
        print()
        
        if board.turn:  # White (human) to move
            print("Your move (white): ", end="")
            user_move = input().strip()
            
            if user_move.lower() == 'quit':
                break
            elif user_move.lower() == 'help':
                print("Enter moves like: e4, Nf3, O-O, Qxd5, etc.")
                continue
            
            try:
                move = board.parse_san(user_move)
                board.push(move)
                print(f"You played: {move}\n")
            except ValueError:
                print("Invalid move! Try again.")
                continue
                
        else:  # Black (engine) to move
            print("Engine thinking...")
            start_time = time.time()
            
            try:
                move = engine.play(board)
                board.push(move)
                
                think_time = time.time() - start_time
                print(f"Engine played: {move} (thought for {think_time:.1f}s)\n")
                
            except Exception as e:
                print(f"Engine error: {e}")
                break
    
    # Game over
    print("Game over!")
    print(f"Result: {board.result()}")
    
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Draw by stalemate")
    elif board.is_insufficient_material():
        print("Draw by insufficient material")
    elif board.is_repetition():
        print("Draw by repetition")


def analyze_position(fen: str, depth: int, model: str):
    """Analyze a specific position."""
    print(f"=== Position Analysis ===\n")
    
    try:
        board = chess.Board(fen)
    except ValueError as e:
        print(f"Invalid FEN: {e}")
        return
    
    print(f"Position: {fen}")
    print(board)
    print()
    
    engine = HybridEngine(model_name=model, max_depth=depth, time_limit=60.0)
    
    print(f"Analyzing with {model} model at depth {depth}...")
    
    try:
        analysis = engine.analyze_position(board, depth=depth)
        
        print(f"\nAnalysis Results:")
        print(f"  Best move: {analysis['best_move']}")
        print(f"  Evaluation: {analysis['evaluation']:.3f}")
        print(f"  Search time: {analysis['search_stats'].time_elapsed:.2f}s")
        print(f"  Nodes searched: {analysis['search_stats'].nodes_searched}")
        print(f"  Neural evaluations: {analysis['search_stats'].evaluations_made}")
        print(f"  Alpha-beta cutoffs: {analysis['search_stats'].alpha_beta_cutoffs}")
        print(f"  Cache hits: {analysis['search_stats'].transformer_cache_hits}")
        
        # Show evaluation rate
        eval_rate = analysis['search_stats'].evaluations_made / analysis['search_stats'].time_elapsed
        print(f"  Evaluation rate: {eval_rate:.1f} evals/sec")
        
    except Exception as e:
        print(f"Analysis error: {e}")


def engine_vs_engine(model1: str, model2: str, depth1: int, depth2: int, games: int):
    """Run engine vs engine matches."""
    print(f"=== Engine vs Engine Match ===\n")
    print(f"{model1} (depth {depth1}) vs {model2} (depth {depth2})")
    print(f"Playing {games} games...\n")
    
    engine1 = HybridEngine(model_name=model1, max_depth=depth1, time_limit=30.0)
    engine2 = HybridEngine(model_name=model2, max_depth=depth2, time_limit=30.0)
    
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}
    
    for game_num in range(games):
        print(f"Game {game_num + 1}/{games}")
        
        board = chess.Board()
        move_count = 0
        
        while not board.is_game_over() and move_count < 100:  # Limit moves
            if board.turn:  # White (engine1)
                try:
                    move = engine1.play(board)
                    board.push(move)
                    print(f"  {move_count//2 + 1}. {move}", end="")
                except Exception as e:
                    print(f"Engine1 error: {e}")
                    break
            else:  # Black (engine2)
                try:
                    move = engine2.play(board)
                    board.push(move)
                    print(f" {move}")
                except Exception as e:
                    print(f"Engine2 error: {e}")
                    break
            
            move_count += 1
        
        result = board.result()
        results[result] += 1
        print(f"  Result: {result}\n")
    
    print("Final Results:")
    print(f"  {model1} wins: {results['1-0']}")
    print(f"  {model2} wins: {results['0-1']}")
    print(f"  Draws: {results['1/2-1/2']}")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Hybrid Chess Engine')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Play command
    play_parser = subparsers.add_parser('play', help='Play against the engine')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a position')
    analyze_parser.add_argument('fen', nargs='?', 
                               default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                               help='FEN position to analyze')
    analyze_parser.add_argument('--depth', type=int, default=4, help='Search depth')
    analyze_parser.add_argument('--model', default='9M', choices=['9M', '136M', '270M'],
                               help='Transformer model to use')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo analysis')
    demo_parser.add_argument('mode', nargs='?', default='positions',
                            choices=['positions', 'depths', 'interactive'],
                            help='Demo mode')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Engine vs engine match')
    match_parser.add_argument('--model1', default='9M', help='First engine model')
    match_parser.add_argument('--model2', default='9M', help='Second engine model')
    match_parser.add_argument('--depth1', type=int, default=3, help='First engine depth')
    match_parser.add_argument('--depth2', type=int, default=4, help='Second engine depth')
    match_parser.add_argument('--games', type=int, default=3, help='Number of games')
    
    args = parser.parse_args()
    
    if args.command == 'play':
        play_game()
        
    elif args.command == 'analyze':
        analyze_position(args.fen, args.depth, args.model)
        
    elif args.command == 'demo':
        # Import and run demo
        from demo import main as demo_main
        sys.argv = ['demo.py', args.mode]  # Fake argv for demo
        demo_main()
        
    elif args.command == 'test':
        # Run test suite
        import subprocess
        test_path = os.path.join(os.path.dirname(__file__), 'tests', 'test_hybrid_engine.py')
        subprocess.run([sys.executable, test_path])
        
    elif args.command == 'match':
        engine_vs_engine(args.model1, args.model2, args.depth1, args.depth2, args.games)
        
    else:
        # Default: show help
        parser.print_help()


if __name__ == '__main__':
    main()
