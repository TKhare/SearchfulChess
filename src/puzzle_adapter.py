"""
Adapter to make the HybridEngine compatible with DeepMind's puzzle evaluation system.

This module provides:
1. An adapter class that implements the Engine protocol
2. Integration with the existing ENGINE_BUILDERS system
3. Registration for puzzle evaluation
"""

import sys
import os
import chess
from typing import Dict, Any

# Add paths for both our engine and DeepMind's code
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../searchless_chess/src'))

from hybrid_engine import HybridEngine
from engines.engine import Engine, AnalysisResult


class HybridEngineAdapter(Engine):
    """
    Adapter that makes HybridEngine compatible with DeepMind's Engine protocol.
    
    This adapter wraps our HybridEngine to work with the puzzle evaluation system
    and other DeepMind infrastructure that expects the Engine interface.
    """
    
    def __init__(
        self,
        model_name: str = '9M_state_value',
        max_depth: int = 4,
        time_limit: float = 5.0,
        use_transposition_table: bool = True,
        enable_move_ordering: bool = True
    ):
        """
        Initialize the hybrid engine adapter.
        
        Args:
            model_name: Model to use (9M_state_value, 9M, 136M, 270M)
            max_depth: Search depth for puzzle solving
            time_limit: Time limit per move (shorter for puzzles)
            use_transposition_table: Enable position caching
            enable_move_ordering: Enable move ordering optimizations
        """
        self.hybrid_engine = HybridEngine(
            model_name=model_name,
            max_depth=max_depth,
            time_limit=time_limit,
            use_transposition_table=use_transposition_table,
            enable_move_ordering=enable_move_ordering
        )
        
        # Store configuration for analysis results
        self.config = {
            'model_name': model_name,
            'max_depth': max_depth,
            'time_limit': time_limit,
            'engine_type': 'HybridEngine'
        }
    
    def play(self, board: chess.Board) -> chess.Move:
        """
        Play the best move in the given position.
        
        This is the main method called by the puzzle evaluation system.
        
        Args:
            board: Chess position to analyze
            
        Returns:
            Best move according to the hybrid engine
        """
        return self.hybrid_engine.play(board)
    
    def analyse(self, board: chess.Board) -> AnalysisResult:
        """
        Analyze a position and return detailed results.
        
        Args:
            board: Chess position to analyze
            
        Returns:
            Analysis results compatible with DeepMind's format
        """
        # Get full analysis from hybrid engine
        analysis = self.hybrid_engine.analyze_position(board)
        
        # Convert to DeepMind format
        return {
            'best_move': analysis['best_move'],
            'evaluation': analysis['evaluation'],
            'search_time': analysis['search_stats'].time_elapsed,
            'nodes_searched': analysis['search_stats'].nodes_searched,
            'neural_evaluations': analysis['search_stats'].evaluations_made,
            'fen': analysis['position_fen'],
            'engine_config': self.config
        }


def create_hybrid_engine_builder(
    model_name: str = '9M_state_value',
    max_depth: int = 4,
    time_limit: float = 5.0
):
    """
    Create a builder function for the hybrid engine.
    
    This creates a function compatible with DeepMind's ENGINE_BUILDERS format.
    
    Args:
        model_name: Model to use
        max_depth: Search depth
        time_limit: Time limit per move
        
    Returns:
        Builder function that creates HybridEngineAdapter instances
    """
    def builder():
        return HybridEngineAdapter(
            model_name=model_name,
            max_depth=max_depth,
            time_limit=time_limit,
            use_transposition_table=True,
            enable_move_ordering=True
        )
    
    return builder


# Pre-configured builders for different hybrid engine configurations
HYBRID_ENGINE_BUILDERS = {
    # State value models (recommended)
    'hybrid_9M_state_value': create_hybrid_engine_builder('9M_state_value', max_depth=4, time_limit=5.0),
    'hybrid_9M_state_value_fast': create_hybrid_engine_builder('9M_state_value', max_depth=3, time_limit=3.0),
    'hybrid_9M_state_value_deep': create_hybrid_engine_builder('9M_state_value', max_depth=6, time_limit=10.0),
    
    # Action value models (fallback)
    'hybrid_9M': create_hybrid_engine_builder('9M', max_depth=4, time_limit=5.0),
    'hybrid_136M': create_hybrid_engine_builder('136M', max_depth=4, time_limit=15.0),
    'hybrid_270M': create_hybrid_engine_builder('270M', max_depth=4, time_limit=30.0),
    
    # Puzzle-optimized configurations
    'hybrid_puzzle_fast': create_hybrid_engine_builder('9M_state_value', max_depth=3, time_limit=2.0),
    'hybrid_puzzle_deep': create_hybrid_engine_builder('9M_state_value', max_depth=5, time_limit=8.0),
}


def register_hybrid_engines():
    """
    Register hybrid engines with DeepMind's ENGINE_BUILDERS system.
    
    This allows hybrid engines to be used in puzzles.py and tournament.py.
    """
    try:
        from engines import constants as engine_constants
        
        # Add our hybrid engines to the existing builders
        engine_constants.ENGINE_BUILDERS.update(HYBRID_ENGINE_BUILDERS)
        
        print(f"Registered {len(HYBRID_ENGINE_BUILDERS)} hybrid engine configurations:")
        for name in HYBRID_ENGINE_BUILDERS.keys():
            print(f"  - {name}")
            
        return True
        
    except ImportError as e:
        print(f"Could not register hybrid engines: {e}")
        return False


def get_available_hybrid_engines():
    """Get list of available hybrid engine configurations."""
    return list(HYBRID_ENGINE_BUILDERS.keys())


if __name__ == '__main__':
    # Test the adapter
    print("Testing HybridEngineAdapter...")
    
    # Create adapter
    adapter = HybridEngineAdapter(model_name='9M_state_value', max_depth=3, time_limit=5.0)
    
    # Test on starting position
    board = chess.Board()
    print(f"Position: {board.fen()}")
    print(board)
    
    # Test play method
    print("\nTesting play() method...")
    move = adapter.play(board)
    print(f"Best move: {move}")
    
    # Test analyse method
    print("\nTesting analyse() method...")
    analysis = adapter.analyse(board)
    print(f"Analysis: {analysis}")
    
    # Test registration
    print("\nTesting engine registration...")
    success = register_hybrid_engines()
    if success:
        print("Available hybrid engines:", get_available_hybrid_engines())
    
    print("\nAdapter test complete!") 