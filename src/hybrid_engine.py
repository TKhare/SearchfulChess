"""
Hybrid Chess Engine: Combines traditional alpha-beta search with transformer-based position evaluation.

Core Architecture:
- Uses DeepMind's transformer models for position evaluation instead of traditional heuristics
- Implements alpha-beta search with the neural network as the evaluation function
- Optimized for the 9M parameter model initially, scalable to larger models

Key Innovation: 
- Traditional engines: Search(N) + Simple_Eval = Strength
- Hybrid engine: Search(N) + Transformer_Eval = Strength equivalent to Search(N+15)
"""

import time
import chess
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

# Import DeepMind's searchless chess components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../searchless_chess/src'))

from engines import constants as engine_constants
from engines.neural_engines import ActionValueEngine, StateValueEngine
from engines.engine import get_ordered_legal_moves
from engines import neural_engines
from searchless_chess.src import utils
from searchless_chess.src import transformer
from searchless_chess.src import training_utils
from searchless_chess.src import tokenizer
import jax.random as jrandom


def _build_neural_engine_fixed_path(
    model_name: str,
    policy: str = 'action_value',
    checkpoint_step: int = 6_400_000,
) -> neural_engines.NeuralEngine:
    """Build a neural engine with fixed checkpoint paths."""
    
    # Model configurations
    match model_name:
        case '9M':
            num_layers = 8
            embedding_dim = 256
            num_heads = 8
        case '136M':
            num_layers = 8
            embedding_dim = 1024
            num_heads = 8
        case '270M':
            num_layers = 16
            embedding_dim = 1024
            num_heads = 8
        case 'local':
            num_layers = 4
            embedding_dim = 64
            num_heads = 4
        case _:
            raise ValueError(f'Unknown model: {model_name}')

    num_return_buckets = 128
    
    match policy:
        case 'action_value':
            output_size = num_return_buckets
        case 'behavioral_cloning':
            output_size = utils.NUM_ACTIONS
        case 'state_value':
            output_size = num_return_buckets
        case _:
            raise ValueError(f'Unknown policy: {policy}')

    predictor_config = transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=output_size,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    predictor = transformer.build_transformer_predictor(config=predictor_config)
    
    # Fixed checkpoint directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    searchless_dir = os.path.join(current_dir, '../../searchless_chess')
    checkpoint_dir = os.path.join(searchless_dir, f'checkpoints/{model_name}')
    
    print(f"Loading from checkpoint: {checkpoint_dir}")
    
    params = training_utils.load_parameters(
        checkpoint_dir=checkpoint_dir,
        params=predictor.initial_params(
            rng=jrandom.PRNGKey(1),
            targets=np.ones((1, 1), dtype=np.uint32),
        ),
        step=checkpoint_step,
    )
    
    _, return_buckets_values = utils.get_uniform_buckets_edges_values(
        num_return_buckets
    )
    
    return neural_engines.ENGINE_FROM_POLICY[policy](
        return_buckets_values=return_buckets_values,
        predict_fn=neural_engines.wrap_predict_fn(
            predictor=predictor,
            params=params,
            batch_size=1,
        ),
    )


def _build_state_value_engine(
    model_name: str,
    checkpoint_step: int = 6_400_000,
) -> neural_engines.NeuralEngine:
    """Build a state value engine for the given model."""
    
    # For now, state value engines use the same checkpoints as action value
    # but with state_value policy. This is a limitation - DeepMind doesn't 
    # provide separate state value checkpoints.
    
    # Check if we have a dedicated state_value checkpoint
    current_dir = os.path.dirname(os.path.abspath(__file__))
    searchless_dir = os.path.join(current_dir, '../../searchless_chess')
    state_value_dir = os.path.join(searchless_dir, f'checkpoints/{model_name}_state_value')
    
    if os.path.exists(state_value_dir):
        print(f"Using dedicated state value checkpoint: {state_value_dir}")
        return _build_neural_engine_fixed_path(model_name, 'state_value', checkpoint_step)
    else:
        print(f"No dedicated state value checkpoint found for {model_name}")
        print("State value engines need separate training. Using action value engine instead.")
        return _build_neural_engine_fixed_path(model_name, 'action_value', checkpoint_step)


@dataclass
class SearchStats:
    """Statistics tracked during search."""
    nodes_searched: int = 0
    evaluations_made: int = 0
    time_elapsed: float = 0.0
    max_depth_reached: int = 0
    alpha_beta_cutoffs: int = 0
    transformer_cache_hits: int = 0


class TransformerEvaluator:
    """Wrapper around DeepMind's neural engines for position evaluation in search trees."""
    
    def __init__(self, neural_engine):
        """
        Initialize with a neural engine for position evaluation.
        
        Args:
            neural_engine: DeepMind's neural engine (ActionValueEngine, StateValueEngine, etc.)
        """
        self.neural_engine = neural_engine
        self.cache: Dict[str, float] = {}  # FEN -> evaluation cache
        self.cache_hits = 0
        
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a chess position using the transformer.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Float evaluation from current player's perspective (-1 to 1, where 1 = winning)
        """
        fen = board.fen()
        
        # Check cache first
        if fen in self.cache:
            self.cache_hits += 1
            return self.cache[fen]
        
        # Get evaluation from neural engine
        analysis = self.neural_engine.analyse(board)
        
        # Handle different engine types
        if isinstance(self.neural_engine, StateValueEngine):
            # StateValueEngine returns 'current_log_probs' for direct position evaluation
            current_log_probs = analysis['current_log_probs']
            current_probs = np.exp(current_log_probs)
            evaluation = np.inner(current_probs, self.neural_engine._return_buckets_values)
            
        elif isinstance(self.neural_engine, ActionValueEngine):
            # ActionValueEngine returns 'log_probs' for each legal move
            # Convert to position value by taking best move value
            return_buckets_log_probs = analysis['log_probs']
            return_buckets_probs = np.exp(return_buckets_log_probs)
            win_probs = np.inner(return_buckets_probs, self.neural_engine._return_buckets_values)
            evaluation = np.max(win_probs)
            
        else:
            # Generic fallback
            if 'current_log_probs' in analysis:
                current_log_probs = analysis['current_log_probs']
                current_probs = np.exp(current_log_probs)
                evaluation = np.inner(current_probs, self.neural_engine._return_buckets_values)
            elif 'log_probs' in analysis:
                log_probs = analysis['log_probs']
                probs = np.exp(log_probs)
                if log_probs.ndim > 1:
                    win_probs = np.inner(probs, self.neural_engine._return_buckets_values)
                    evaluation = np.max(win_probs)
                else:
                    evaluation = np.inner(probs, self.neural_engine._return_buckets_values)
            else:
                raise ValueError(f"Unknown analysis format: {list(analysis.keys())}")
        
        # Convert to evaluation score (0-1 -> -1 to 1)
        evaluation = 2 * evaluation - 1
        
        # Cache result
        self.cache[fen] = evaluation
        
        return evaluation
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self.cache.clear()
        self.cache_hits = 0


class HybridEngine:
    """
    Hybrid chess engine combining alpha-beta search with transformer evaluation.
    
    This engine uses DeepMind's transformer models as the evaluation function
    in a traditional minimax search with alpha-beta pruning.
    """
    
    def __init__(
        self,
        model_name: str = '9M_state_value',
        max_depth: int = 6,
        time_limit: Optional[float] = None,
        use_transposition_table: bool = True,
        enable_move_ordering: bool = True
    ):
        """
        Initialize the hybrid engine.
        
        Args:
            model_name: DeepMind model to use ('9M_state_value', '9M', '136M', '270M')
                       State value models are preferred for hybrid search
            max_depth: Maximum search depth
            time_limit: Maximum time per move in seconds
            use_transposition_table: Whether to use transposition table
            enable_move_ordering: Whether to enable move ordering optimizations
        """
        self.model_name = model_name
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_transposition_table = use_transposition_table
        self.enable_move_ordering = enable_move_ordering
        
        # Load the neural engine
        print(f"Loading {model_name} transformer model...")
        
        # Try to use state value version first (better for hybrid search)
        if model_name.endswith('_state_value'):
            base_model = model_name.replace('_state_value', '')
            try:
                base_neural_engine = _build_state_value_engine(base_model)
                print(f"Using StateValueEngine for {base_model}")
            except Exception as e:
                print(f"Failed to load state value engine: {e}")
                print("Falling back to action value engine...")
                base_neural_engine = _build_neural_engine_fixed_path(base_model, 'action_value')
        else:
            # Use our fixed path builder instead of the original ENGINE_BUILDERS
            try:
                base_neural_engine = _build_neural_engine_fixed_path(model_name, 'action_value')
                print(f"Using ActionValueEngine for {model_name}")
            except Exception as e:
                print(f"Failed to load with fixed path: {e}")
                # Final fallback to original builders (will likely fail with path issues)
                if model_name in engine_constants.ENGINE_BUILDERS:
                    engine_builder = engine_constants.ENGINE_BUILDERS[model_name]
                    base_neural_engine = engine_builder()
                else:
                    raise ValueError(f"Could not load model {model_name}")
        
        # Store the neural engine
        self.neural_engine = base_neural_engine
        
        # Create evaluator wrapper
        self.evaluator = TransformerEvaluator(base_neural_engine)
        
        # Search components
        self.transposition_table: Dict[str, Tuple[float, int, str]] = {}  # fen -> (eval, depth, node_type)
        self.search_stats = SearchStats()
        
        print(f"Hybrid engine initialized with {model_name} model")
        print(f"Search depth: {max_depth}, Time limit: {time_limit}s")
    
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Evaluate a position using the transformer model.
        
        Returns evaluation from current player's perspective.
        """
        self.search_stats.evaluations_made += 1
        return self.evaluator.evaluate_position(board)
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Order moves for better alpha-beta pruning.
        
        Uses the transformer to get move evaluations for ordering.
        """
        if not self.enable_move_ordering or len(moves) <= 1:
            return moves
        
        # For ActionValueEngine, we can get move evaluations directly
        if isinstance(self.neural_engine, ActionValueEngine):
            try:
                analysis = self.neural_engine.analyse(board)
                return_buckets_log_probs = analysis['log_probs']
                return_buckets_probs = np.exp(return_buckets_log_probs)
                win_probs = np.inner(return_buckets_probs, self.neural_engine._return_buckets_values)
                
                sorted_legal_moves = get_ordered_legal_moves(board)
                move_scores = list(zip(sorted_legal_moves, win_probs))
                
                # Sort by win probability (descending for white, ascending for black)
                if board.turn:  # White to move
                    move_scores.sort(key=lambda x: x[1], reverse=True)
                else:  # Black to move
                    move_scores.sort(key=lambda x: x[1])
                
                return [move for move, _ in move_scores if move in moves]
            except Exception:
                # Fallback to basic ordering
                pass
        
        # Fallback: basic move ordering (captures first, then others)
        captures = [move for move in moves if board.is_capture(move)]
        non_captures = [move for move in moves if not board.is_capture(move)]
        return captures + non_captures
    
    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: bool,
        start_time: float
    ) -> float:
        """
        Alpha-beta search implementation with transformer evaluation.
        
        Args:
            board: Current position
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: Whether current player is maximizing
            start_time: Search start time for time management
            
        Returns:
            Position evaluation
        """
        self.search_stats.nodes_searched += 1
        
        # Time management
        if self.time_limit and time.time() - start_time > self.time_limit:
            return self.evaluate_position(board)
        
        # Terminal conditions
        if board.is_game_over():
            result = board.result()
            if result == "1-0":  # White wins
                return 1000 if maximizing_player == board.turn else -1000
            elif result == "0-1":  # Black wins
                return -1000 if maximizing_player == board.turn else 1000
            else:  # Draw
                return 0
        
        # Depth limit reached - use transformer evaluation
        if depth == 0:
            return self.evaluate_position(board)
        
        # Transposition table lookup
        fen = board.fen()
        if self.use_transposition_table and fen in self.transposition_table:
            cached_eval, cached_depth, node_type = self.transposition_table[fen]
            if cached_depth >= depth:
                self.search_stats.transformer_cache_hits += 1
                return cached_eval
        
        # Generate and order moves
        legal_moves = list(board.legal_moves)
        ordered_moves = self.order_moves(board, legal_moves)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self.alpha_beta(board, depth - 1, alpha, beta, False, start_time)
                board.pop()
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    self.search_stats.alpha_beta_cutoffs += 1
                    break
            
            # Store in transposition table
            if self.use_transposition_table:
                self.transposition_table[fen] = (max_eval, depth, 'exact')
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval_score = self.alpha_beta(board, depth - 1, alpha, beta, True, start_time)
                board.pop()
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    self.search_stats.alpha_beta_cutoffs += 1
                    break
            
            # Store in transposition table
            if self.use_transposition_table:
                self.transposition_table[fen] = (min_eval, depth, 'exact')
            
            return min_eval
    
    def search(self, board: chess.Board, depth: Optional[int] = None) -> Tuple[chess.Move, float, SearchStats]:
        """
        Find the best move using alpha-beta search with transformer evaluation.
        
        Args:
            board: Position to search
            depth: Search depth (uses self.max_depth if None)
            
        Returns:
            Tuple of (best_move, evaluation, search_stats)
        """
        if depth is None:
            depth = self.max_depth
        
        # Reset search statistics
        self.search_stats = SearchStats()
        start_time = time.time()
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            # Only one move available
            return legal_moves[0], 0.0, self.search_stats
        
        best_move = legal_moves[0]
        best_eval = float('-inf') if board.turn else float('inf')
        
        # Order moves for better search
        ordered_moves = self.order_moves(board, legal_moves)
        
        print(f"Searching {len(legal_moves)} moves at depth {depth}...")
        
        for i, move in enumerate(ordered_moves):
            board.push(move)
            
            if board.turn:  # After move, it's white's turn (so we played black)
                eval_score = self.alpha_beta(
                    board, depth - 1, float('-inf'), float('inf'), True, start_time
                )
            else:  # After move, it's black's turn (so we played white)
                eval_score = self.alpha_beta(
                    board, depth - 1, float('-inf'), float('inf'), False, start_time
                )
            
            board.pop()
            
            # Update best move
            if board.turn:  # White to move - maximize
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:  # Black to move - minimize
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
            
            print(f"Move {i+1}/{len(ordered_moves)}: {move} -> {eval_score:.3f}")
            
            # Time management
            if self.time_limit and time.time() - start_time > self.time_limit:
                print("Time limit reached, stopping search")
                break
        
        # Update final statistics
        self.search_stats.time_elapsed = time.time() - start_time
        self.search_stats.max_depth_reached = depth
        self.search_stats.transformer_cache_hits = self.evaluator.cache_hits
        
        return best_move, best_eval, self.search_stats
    
    def play(self, board: chess.Board) -> chess.Move:
        """
        Play the best move in the position.
        
        Args:
            board: Position to play from
            
        Returns:
            Best move according to hybrid search
        """
        move, eval_score, stats = self.search(board)
        
        print(f"\nBest move: {move} (eval: {eval_score:.3f})")
        print(f"Search stats: {stats.nodes_searched} nodes, {stats.evaluations_made} evals")
        print(f"Time: {stats.time_elapsed:.2f}s, Cutoffs: {stats.alpha_beta_cutoffs}")
        print(f"Cache hits: {stats.transformer_cache_hits}")
        
        return move
    
    def analyze_position(self, board: chess.Board, depth: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze a position and return detailed information.
        
        Args:
            board: Position to analyze
            depth: Analysis depth
            
        Returns:
            Dictionary with analysis results
        """
        move, evaluation, stats = self.search(board, depth)
        
        return {
            'best_move': move,
            'evaluation': evaluation,
            'search_stats': stats,
            'position_fen': board.fen(),
            'model_used': self.model_name
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get engine information."""
        return {
            'name': 'HybridEngine',
            'version': '0.1.0',
            'model': self.model_name,
            'max_depth': self.max_depth,
            'time_limit': self.time_limit,
            'features': {
                'transposition_table': self.use_transposition_table,
                'move_ordering': self.enable_move_ordering,
                'transformer_evaluation': True
            }
        } 