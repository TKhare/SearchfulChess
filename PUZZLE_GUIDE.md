# Hybrid Engine Puzzle Evaluation Guide

This guide explains how to run chess puzzle evaluation using our hybrid engine with DeepMind's puzzle dataset.

## Overview

Our hybrid engine can now be evaluated on the same chess puzzles used by DeepMind to benchmark their neural engines. This allows direct comparison of:

- **Hybrid engines** (our search + transformer evaluation)
- **Pure neural engines** (DeepMind's searchless models)  
- **Traditional engines** (Stockfish, Leela Chess Zero)

## Setup

### 1. Download Puzzle Data

First, download DeepMind's puzzle dataset:

```bash
cd ../searchless_chess/data
wget https://storage.googleapis.com/searchless_chess/data/puzzles.csv
```

The puzzle dataset contains thousands of chess puzzles from Lichess with:
- **PGN**: The game leading up to the puzzle position
- **Moves**: The solution sequence
- **Rating**: Difficulty rating (higher = harder)
- **Themes**: Puzzle themes (tactics, endgame, etc.)

### 2. Test Setup

Verify everything works:

```bash
cd engine/
python test_puzzle_setup.py
```

This will test:
- ✓ All imports work correctly
- ✓ Hybrid engine can be created  
- ✓ Puzzle adapter functions
- ✓ Engine registration works
- ✓ Puzzle data is available

## Running Puzzle Evaluation

### Basic Usage

Evaluate 10 puzzles with the hybrid engine:

```bash
python puzzle_runner.py --engine hybrid_9M_state_value --num_puzzles 10
```

### Available Engines

List all available engines:

```bash
python puzzle_runner.py --list_engines
```

**Hybrid Engines:**
- `hybrid_9M_state_value` - Recommended default (state value model)
- `hybrid_9M_state_value_fast` - Faster, shallower search  
- `hybrid_9M_state_value_deep` - Deeper search for accuracy
- `hybrid_9M` - Action value model (fallback)
- `hybrid_136M` - Larger model, slower but stronger
- `hybrid_270M` - Largest model, maximum strength
- `hybrid_puzzle_fast` - Optimized for fast puzzle solving
- `hybrid_puzzle_deep` - Optimized for difficult puzzles

**DeepMind Neural Engines:**
- `9M`, `136M`, `270M` - Pure neural engines (no search)
- `stockfish`, `stockfish_all_moves` - Traditional engines
- `leela_chess_zero_*` - Leela Chess Zero variants

### Engine Comparison

Compare multiple engines on the same puzzles:

```bash
# Compare hybrid vs neural
python puzzle_runner.py --compare hybrid_9M_state_value 9M --num_puzzles 20

# Compare different hybrid configurations  
python puzzle_runner.py --compare hybrid_9M_state_value_fast hybrid_9M_state_value_deep --num_puzzles 20

# Compare hybrid vs traditional
python puzzle_runner.py --compare hybrid_9M_state_value stockfish --num_puzzles 20
```

### Large-Scale Evaluation

For comprehensive evaluation:

```bash
# Evaluate 100 puzzles (typical benchmark)
python puzzle_runner.py --engine hybrid_9M_state_value --num_puzzles 100

# Full dataset evaluation (thousands of puzzles)
python puzzle_runner.py --engine hybrid_9M_state_value --num_puzzles 10000
```

## Understanding Results

### Example Output

```
============================================================
Puzzle   0: ✓ (Rating: 1285, Time: 2.34s)
Puzzle   1: ✗ (Rating: 1854, Time: 4.12s)  
Puzzle   2: ✓ (Rating: 1456, Time: 1.89s)
...
============================================================
RESULTS SUMMARY:
Engine: hybrid_9M_state_value
Puzzles solved: 8/10 (80.0%)
Average time per puzzle: 2.45s
Total time: 24.5s
Average puzzle rating: 1567
```

### Key Metrics

- **Accuracy**: Percentage of puzzles solved correctly
- **Average Time**: Time per puzzle (lower is better for speed)
- **Average Rating**: Difficulty of puzzles attempted
- **Total Time**: Overall evaluation time

### Comparison Format

```
===============================================================================
COMPARISON SUMMARY
===============================================================================
Engine                    Solved     Accuracy   Avg Time   Total Time
-------------------------------------------------------------------------------
hybrid_9M_state_value     8/10       80.0%      2.45s      24.5s
9M                        6/10       60.0%      0.12s      1.2s
stockfish                 9/10       90.0%      0.05s      0.5s
```

## Expected Performance

### Puzzle Solving Accuracy

Based on our hybrid approach, expected performance:

| Engine Type | Easy Puzzles | Medium Puzzles | Hard Puzzles | Overall |
|-------------|--------------|----------------|--------------|---------|
| Hybrid 9M   | ~85%         | ~70%          | ~50%         | ~75%    |
| Neural 9M   | ~70%         | ~55%          | ~35%         | ~60%    |
| Stockfish   | ~90%         | ~85%          | ~75%         | ~85%    |

### Search Advantage

The hybrid engine should show particular strength in:

1. **Tactical Puzzles**: Where search depth helps find combinations
2. **Medium Difficulty**: Where neural evaluation + search is optimal
3. **Complex Positions**: Where pure pattern recognition isn't enough

## Configuration Options

### Hybrid Engine Parameters

Adjust engine parameters for different use cases:

```python
# Fast puzzle solving (tournaments)
HybridEngineAdapter(
    model_name='9M_state_value',
    max_depth=3,              # Shallow search
    time_limit=2.0,           # Fast decisions
)

# Accurate puzzle solving (analysis)  
HybridEngineAdapter(
    model_name='9M_state_value',
    max_depth=6,              # Deep search
    time_limit=10.0,          # More thinking time
)

# Maximum strength (research)
HybridEngineAdapter(
    model_name='270M',        # Largest model
    max_depth=8,              # Very deep search  
    time_limit=30.0,          # Extensive analysis
)
```

### Optimization for Puzzles

Puzzles benefit from:
- **Deeper search** (tactical combinations)
- **State value models** (direct position evaluation)
- **Move ordering** (find key moves faster)
- **Reasonable time limits** (balance speed vs accuracy)

## Troubleshooting

### Common Issues

**Import Error**: Make sure paths are set correctly
```bash
export PYTHONPATH=/path/to/LLMChessEngine/engine/src:/path/to/searchless_chess/src
```

**Model Loading Error**: Verify checkpoint directory
```bash
ls ../searchless_chess/checkpoints/9M_state_value/
```

**Memory Issues**: Use smaller models or reduce batch size
```python
# In hybrid_engine.py, reduce batch_size in wrap_predict_fn
batch_size=1  # Instead of 32
```

**Slow Performance**: Use faster configurations
```bash
python puzzle_runner.py --engine hybrid_puzzle_fast --num_puzzles 10
```

### Performance Tips

1. **Use State Value Models**: Faster and more appropriate for search
2. **Adjust Search Depth**: Balance accuracy vs speed
3. **Enable Caching**: Transposition tables help with repeated positions
4. **Batch Evaluation**: Run many puzzles at once for better statistics

## Research Applications

### Validation Experiments

Use puzzle evaluation to validate our key hypotheses:

**1. Search Enhancement**
```bash
# Compare search depths
python puzzle_runner.py --compare hybrid_9M_state_value_fast hybrid_9M_state_value_deep --num_puzzles 100
```

**2. Neural vs Hybrid**
```bash
# Test if search improves neural evaluation
python puzzle_runner.py --compare 9M hybrid_9M_state_value --num_puzzles 100
```

**3. Model Scaling**
```bash
# Test larger models
python puzzle_runner.py --compare hybrid_9M hybrid_136M hybrid_270M --num_puzzles 50
```

### Data Collection

Puzzle results can be used for:
- **Elo Estimation**: Convert accuracy to Elo ratings
- **Position Analysis**: Identify where hybrid excels/struggles  
- **Search Efficiency**: Measure nodes per correct solution
- **Time Scaling**: Analyze speed vs accuracy tradeoffs

## Integration with DeepMind Code

Our puzzle runner integrates seamlessly with DeepMind's infrastructure:

- **Same Format**: Uses identical puzzle evaluation logic
- **Compatible Results**: Can be compared directly with their benchmarks
- **Engine Protocol**: Implements their Engine interface
- **Registration System**: Adds to their ENGINE_BUILDERS

This allows our hybrid approach to be evaluated using their established methodology and compared fairly with their results.

## Next Steps

After puzzle evaluation, consider:

1. **Tournament Play**: Use `tournament.py` for engine vs engine games
2. **Custom Puzzles**: Create specialized puzzle sets for specific themes
3. **Optimization**: Tune search parameters based on puzzle results
4. **Analysis**: Deep dive into failed puzzles to improve evaluation

The puzzle evaluation provides crucial validation of our hybrid approach and direct comparison with state-of-the-art chess AI systems. 