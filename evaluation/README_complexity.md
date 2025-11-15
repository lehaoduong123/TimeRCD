# Computational Complexity Analysis

This directory contains scripts for analyzing the computational complexity (time and space) of TimeRCD.

## Script: `complexity_analysis.py`

This script profiles TimeRCD's computational complexity by measuring:
- **Time complexity**: How inference time scales with sequence length
- **Space complexity**: How memory usage scales with sequence length
- **Component breakdown**: Time spent in encoder, reconstruction head, and anomaly head
- **Training cost**: Synthetic training-step time (forward + backward)
- **Model capacity**: Total number of learnable parameters

## Usage

### Basic Usage

```bash
python evaluation/complexity_analysis.py
```

This will run with default settings:
- Sequence lengths: [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
- Single feature (univariate)
- Batch size: 1
- 10 runs per configuration

### Custom Configuration

```bash
python evaluation/complexity_analysis.py \
    --seq_lengths 5000 10000 15000 20000 25000 30000 \
    --feature_counts 1 8 16 \
    --batch_size 4 \
    --num_runs 20 \
    --device cuda \
    --output_dir evaluation/complexity_results
```

### Arguments

- `--seq_lengths`: List of sequence lengths to test (default: [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000])
- `--num_features`: Number of features (default: 1). Ignored if `--feature_counts` is provided.
- `--feature_counts`: Optional list of feature counts to profile. When specified, the script runs the full analysis for each count (including training timings) and stores outputs in subdirectories such as `evaluation/complexity_results/features_8`.
- `--batch_size`: Batch size for testing (default: 1)
- `--num_runs`: Number of runs per configuration for averaging (default: 10)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')
- `--output_dir`: Directory to save plots and reports (default: 'evaluation/complexity_plots')

## Output

The script generates:

1. **`complexity_analysis.png`**: Comprehensive visualization with 4 subplots:
   - Time complexity vs sequence length
   - Space complexity vs sequence length
   - Normalized time per element
   - Component time breakdown (pie chart)

2. **`scaling_analysis.png`**: Detailed scaling analysis with fitted complexity curve

3. **`complexity_report_features_{k}.json`**: Detailed JSON report (per feature count `k`) with:
   - Per-configuration metrics
   - Scaling factors
   - Estimated complexity classes (O(n^k))
   - Training-step timings and parameter counts per configuration

## Example Output

```
TimeRCD Computational Complexity Analysis
============================================================
Device: cuda
Sequence lengths: [1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000]
Features: 1, Batch size: 1
Runs per config: 10
============================================================
Analyzing scaling behavior with 8 sequence lengths...
Features: 1, Batch size: 1
------------------------------------------------------------
Testing sequence length: 1000... ✓ (Time: 12.34ms, Memory: 45.67MB)
Testing sequence length: 2000... ✓ (Time: 23.45ms, Memory: 67.89MB)
...
```

## Understanding the Results

### Time Complexity

The script measures:
- **Total time**: End-to-end inference time
- **Encoder time**: Time spent in the TimeSeriesEncoder
- **Reconstruction head time**: Time for reconstruction head forward pass
- **Anomaly head time**: Time for anomaly detection head forward pass
- **Training step time**: Approximate duration of a full training step (forward, loss, backward) with synthetic data

### Space Complexity

- **Peak memory**: Maximum GPU/CPU memory used during inference
- Measured in MB (megabytes)
- **Model parameters**: Reported for each configuration to contextualize capacity

### Scaling Analysis

The script fits a curve to determine the complexity class:
- **O(n)**: Linear scaling (ideal)
- **O(n log n)**: Near-linear scaling (good)
- **O(n²)**: Quadratic scaling (attention mechanism)

## Integration with Other Analysis

This script can be used alongside:
- `evaluation/efficiency_analysis.py` (if created) - for runtime efficiency
- Statistical testing scripts - for performance validation
- Length analysis scripts - for generalization studies

## Notes

- The script uses warmup runs to ensure accurate timing
- GPU synchronization is used when CUDA is available
- Memory measurements are approximate and may vary based on system state
- For CPU profiling, ensure sufficient RAM is available

