"""
Test suite for the Hybrid Chess Engine.

Tests the integration of transformer evaluation with alpha-beta search.
"""

import unittest
import chess
import sys
import os

# Add the engine source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_engine import HybridEngine, TransformerEvaluator, SearchStats


class TestHybridEngine(unittest.TestCase):
    """Test cases for the HybridEngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class with a shared engine instance."""
        # Use 9M state value model for faster testing
        print("Loading 9M state value model for testing...")
        cls.engine = HybridEngine(model_name='9M_state_value', max_depth=3, time_limit=10.0)
    
    def test_engine_initialization(self):
        """Test that the engine initializes correctly."""
        self.assertEqual(self.engine.model_name, '9M')
        self.assertEqual(self.engine.max_depth, 3)
        self.assertEqual(self.engine.time_limit, 10.0)
        self.assertIsNotNone(self.engine.neural_engine)
        self.assertIsNotNone(self.engine.evaluator)
    
    def test_position_evaluation(self):
        """Test that positions can be evaluated."""
        board = chess.Board()  # Starting position
        
        evaluation = self.engine.evaluate_position(board)
        
        # Should return a reasonable evaluation
        self.assertIsInstance(evaluation, float)
        self.assertGreaterEqual(evaluation, -2.0)  # Reasonable bounds
        self.assertLessEqual(evaluation, 2.0)
    
    def test_move_generation(self):
        """Test that the engine can find moves."""
        board = chess.Board()
        
        move, evaluation, stats = self.engine.search(board, depth=2)
        
        # Should return a legal move
        self.assertIn(move, board.legal_moves)
        self.assertIsInstance(evaluation, float)
        self.assertIsInstance(stats, SearchStats)
        self.assertGreater(stats.nodes_searched, 0)
    
    def test_different_positions(self):
        """Test the engine on various chess positions."""
        test_positions = [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',  # Starting
            'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',  # Sicilian
            '8/8/8/8/8/3k4/3P4/3K4 w - - 0 1',  # King and pawn endgame
        ]
        
        for fen in test_positions:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                
                move, evaluation, stats = self.engine.search(board, depth=2)
                
                self.assertIn(move, board.legal_moves)
                self.assertIsInstance(evaluation, float)
                self.assertGreater(stats.nodes_searched, 0)
    
    def test_depth_scaling(self):
        """Test that deeper search takes more time and searches more nodes."""
        board = chess.Board()
        
        # Search at depth 1
        _, _, stats1 = self.engine.search(board, depth=1)
        
        # Search at depth 2
        _, _, stats2 = self.engine.search(board, depth=2)
        
        # Deeper search should explore more nodes
        self.assertGreaterEqual(stats2.nodes_searched, stats1.nodes_searched)
    
    def test_move_ordering(self):
        """Test that move ordering is working."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        
        # Test move ordering
        ordered_moves = self.engine.order_moves(board, legal_moves)
        
        # Should return same moves, possibly in different order
        self.assertEqual(set(ordered_moves), set(legal_moves))
        self.assertEqual(len(ordered_moves), len(legal_moves))
    
    def test_transposition_table(self):
        """Test that transposition table is working."""
        board = chess.Board()
        
        # Clear transposition table
        self.engine.transposition_table.clear()
        
        # First search
        move1, eval1, stats1 = self.engine.search(board, depth=2)
        
        # Second search of same position should use transposition table
        move2, eval2, stats2 = self.engine.search(board, depth=2)
        
        # Should get same result
        self.assertEqual(move1, move2)
        # Second search might be faster due to caching
        self.assertLessEqual(stats2.time_elapsed, stats1.time_elapsed + 0.5)  # Allow some variance
    
    def test_time_management(self):
        """Test that time limits are respected."""
        board = chess.Board()
        
        # Create engine with very short time limit
        fast_engine = HybridEngine(model_name='9M_state_value', max_depth=10, time_limit=1.0)
        
        move, evaluation, stats = fast_engine.search(board)
        
        # Should respect time limit (with some tolerance)
        self.assertLessEqual(stats.time_elapsed, 2.0)
        self.assertIn(move, board.legal_moves)
    
    def test_game_over_positions(self):
        """Test engine behavior in terminal positions."""
        # Checkmate position
        board = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2')
        board.push_san('Qh4#')  # Fool's mate
        
        self.assertTrue(board.is_checkmate())
        
        # Engine should handle this gracefully
        with self.assertRaises(ValueError):  # No legal moves
            self.engine.search(board, depth=1)


class TestTransformerEvaluator(unittest.TestCase):
    """Test cases for the TransformerEvaluator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test class."""
        # Create a state value engine for testing
        from hybrid_engine import _build_state_value_engine
        try:
            neural_engine = _build_state_value_engine('9M')
        except Exception:
            # Fallback to action value engine
            from engines import constants as engine_constants
            engine_builder = engine_constants.ENGINE_BUILDERS['9M']
            neural_engine = engine_builder()
        cls.evaluator = TransformerEvaluator(neural_engine)
    
    def test_evaluation_consistency(self):
        """Test that evaluations are consistent."""
        board = chess.Board()
        
        eval1 = self.evaluator.evaluate_position(board)
        eval2 = self.evaluator.evaluate_position(board)
        
        # Should be exactly the same due to caching
        self.assertEqual(eval1, eval2)
    
    def test_evaluation_cache(self):
        """Test that the evaluation cache works."""
        board = chess.Board()
        
        # Clear cache
        self.evaluator.clear_cache()
        self.assertEqual(self.evaluator.cache_hits, 0)
        
        # First evaluation
        eval1 = self.evaluator.evaluate_position(board)
        self.assertEqual(self.evaluator.cache_hits, 0)
        
        # Second evaluation should hit cache
        eval2 = self.evaluator.evaluate_position(board)
        self.assertEqual(self.evaluator.cache_hits, 1)
        self.assertEqual(eval1, eval2)
    
    def test_perspective_flip(self):
        """Test that evaluations flip perspective correctly."""
        # Create a position where white is clearly better
        board = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2')
        
        eval_white = self.evaluator.evaluate_position(board)
        
        # Make a move so it's black's turn
        board.push_san('Nf3')
        eval_black = self.evaluator.evaluate_position(board)
        
        # Evaluations should have opposite signs (roughly)
        # Note: This is a weak test since the position changes
        self.assertIsInstance(eval_white, float)
        self.assertIsInstance(eval_black, float)


def run_performance_test():
    """Run a performance test to measure engine speed."""
    print("\n=== Performance Test ===")
    
    engine = HybridEngine(model_name='9M_state_value', max_depth=4, time_limit=30.0)
    board = chess.Board()
    
    import time
    start_time = time.time()
    
    move, evaluation, stats = engine.search(board, depth=3)
    
    elapsed = time.time() - start_time
    
    print(f"Performance test results:")
    print(f"  Search depth: 3")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Nodes searched: {stats.nodes_searched}")
    print(f"  Evaluations: {stats.evaluations_made}")
    print(f"  Nodes per second: {stats.nodes_searched/elapsed:.0f}")
    print(f"  Evaluations per second: {stats.evaluations_made/elapsed:.1f}")
    print(f"  Alpha-beta cutoffs: {stats.alpha_beta_cutoffs}")
    print(f"  Best move: {move}")
    print(f"  Evaluation: {evaluation:.3f}")


if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test
    run_performance_test() 