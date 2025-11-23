# Profiling Improvements and Issues Analysis

## Summary

This document explains the improvements made to memory and time tracking in `complexity_analysis.py` and discusses potential issues with the reported efficiency comparison results.

## Improvements Made

### 1. Enhanced Time Tracking

**Previous Implementation:**
- Used `time.time()` for timing (less precise)
- Basic CUDA synchronization
- No statistical measures (std deviation)

**New Implementation:**
- **CPU Timing**: Uses `time.perf_counter()` for higher precision (nanosecond-level)
- **GPU Timing**: Uses `torch.cuda.Event` for accurate GPU timing when available
- **Statistical Measures**: Reports standard deviation across multiple runs
- **Better Synchronization**: Ensures all GPU operations complete before measuring

```python
# Example: Using CUDA events for accurate GPU timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
# ... operations ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

### 2. Comprehensive Memory Tracking

**Previous Implementation:**
- Only tracked PyTorch GPU memory via `torch.cuda.max_memory_allocated()`
- Missing memory for models using CPU or non-PyTorch frameworks

**New Implementation:**
- **GPU Memory**: Tracks PyTorch GPU memory (unchanged but more accurate)
- **System Memory**: Uses `psutil` to track process RSS memory
- **Memory Delta**: Calculates memory delta from baseline for accurate measurement
- **Separate Reporting**: Reports both GPU and system memory separately

```python
# Example: Comprehensive memory tracking
peak_gpu_mem, peak_system_mem = get_peak_memory_mb(include_system=True)
system_memory_delta = peak_system_mem - baseline_system_mem
```

### 3. Better Baseline Comparisons

**Improvements:**
- Models like Chronos that use AutoGluon may run on CPU - now properly tracked
- Separate handling for models that include training time in their API
- More accurate device placement tracking

## Issues with Reported Results

Based on the analysis of the code, here are the **potential issues** with the efficiency comparison table:

### Issue 1: Chronos Memory Measurement (CRITICAL)

**Reported**: 0.005-0.024 MB  
**Expected**: Much higher (200M parameter model)

**Root Cause:**
- Chronos uses AutoGluon's `TimeSeriesPredictor` which:
  - May run on CPU instead of GPU
  - Manages memory outside PyTorch's tracking
  - Creates/fits models in temporary directories

**Solution:**
The new implementation tracks system memory separately, which should capture Chronos's actual memory usage. However, note that:
- If Chronos runs on CPU, GPU memory will be near-zero (expected)
- System memory tracking will show the actual memory used

### Issue 2: Chronos Time Measurement

**Reported**: 263-1041 ms  
**Expected**: Should be slower if it includes training

**Root Cause:**
- Chronos's `fit()` method includes **model training**, not just inference
- This is fundamentally different from TimeRCD which only does inference
- For 1000 timesteps with window_size=100, it fits 10 separate models

**Solution:**
The code now documents this clearly, but the comparison may still be unfair:
- TimeRCD: Inference only (model pre-loaded)
- Chronos: Training + inference per chunk

**Recommendation**: Report Chronos separately or note that it includes training time.

### Issue 3: Inconsistent Device Placement

**Potential Issue:**
- Different models may run on different devices (GPU vs CPU)
- This can significantly affect both time and memory measurements

**Solution:**
The new code:
- Explicitly tracks device placement
- Uses system memory tracking for CPU-based operations
- Documents device usage for each model

### Issue 4: Chunking Overhead

**Potential Issue:**
- Baseline models use chunking (window sizes), which adds overhead
- TimeRCD processes full sequences without chunking
- This overhead is included in the measurements (intentional for fair comparison)

**Status**: This is documented and intentional, but should be noted in publications.

## Recommended Best Practices

### 1. When Comparing Models

Always report:
- **Inference vs Training**: Separate inference-only comparisons from training+inference
- **Device**: GPU vs CPU usage
- **Memory Type**: GPU memory vs system memory
- **Statistics**: Mean ± std deviation across multiple runs
- **Warmup**: Number of warmup runs excluded from measurements

### 2. For Chronos Specifically

- **Memory**: Use system memory tracking (RSS) not GPU memory
- **Time**: Note that it includes training time, not just inference
- **Fairness**: Consider comparing inference-only if possible

### 3. For Reporting

Include in tables/figures:
- Standard deviation for timing
- Both GPU and system memory (if applicable)
- Device used (GPU vs CPU)
- Number of runs for statistics
- Notes about what's included (inference-only vs training+inference)

## Usage Example

```python
# Run with improved profiling
profiler = ComplexityProfiler(config, device="cuda", model_name="timercd")
results = profiler.analyze_scaling(
    seq_lengths=[1000, 2000, 5000],
    num_features=1,
    batch_size=1,
    num_runs=10  # More runs for better statistics
)

# Results now include:
# - std_time_ms: Standard deviation of timing
# - peak_system_memory_mb: System memory tracking
# - More accurate GPU timing with CUDA events
```

## Testing the Improvements

To verify the improvements work correctly:

```bash
# Test TimeRCD with improved profiling
python evaluation/complexity_analysis.py \
    --model_name timercd \
    --seq_lengths 1000 2000 \
    --num_runs 10

# Test a baseline (e.g., Chronos) with system memory tracking
python evaluation/complexity_analysis.py \
    --model_name chronos \
    --seq_lengths 1000 2000 \
    --num_runs 10
```

Check the output for:
- Standard deviation in timing (±X.XXms)
- System memory reporting (sys: X.XXMB)
- More consistent measurements across runs

## Handling Concurrent Scripts

### Impact of Running Multiple Scripts Simultaneously

**Yes, running multiple scripts can affect memory measurements!** Here's why and how the code handles it:

#### GPU Memory Issues

1. **Shared GPU Memory**: If multiple scripts use the same GPU:
   - `torch.cuda.max_memory_allocated()` tracks **per-process** memory, but:
   - Total GPU memory is shared across processes
   - Memory fragmentation can occur
   - Other processes may not release memory properly

2. **Timing Interference**: 
   - GPU context switching between processes
   - Competition for GPU resources
   - Can cause inconsistent timing measurements

#### System Memory Issues

1. **Per-Process Tracking**: `psutil.Process(os.getpid())` tracks **only the current process**, so:
   - ✅ System memory tracking should be OK
   - ❌ But overall system performance can still affect measurements

2. **CPU Context Switching**: Multiple processes can cause:
   - Timing variations
   - CPU competition affecting CPU-based models (like Chronos on CPU)

### Safeguards Added

The improved code now includes:

1. **GPU Isolation Checking**:
   - Detects if other processes are using the GPU
   - Warns if GPU memory is already allocated
   - Reports initial GPU memory state

2. **Stable Baseline Measurement**:
   - Multiple baseline readings for system memory
   - Delays to ensure memory cleanup
   - Better garbage collection

3. **Initial State Reporting**:
   - Shows GPU memory before profiling starts
   - Warns if memory is already in use
   - Suggests running in isolation

### Best Practices for Accurate Results

1. **Run in Isolation** (Recommended):
   ```bash
   # Stop other GPU processes first
   nvidia-smi  # Check what's running
   # Kill other processes if needed
   
   # Run profiling alone
   python evaluation/complexity_analysis.py --model_name timercd
   ```

2. **Use GPU Isolation**:
   ```bash
   # Use a dedicated GPU
   CUDA_VISIBLE_DEVICES=0 python evaluation/complexity_analysis.py --model_name timercd
   CUDA_VISIBLE_DEVICES=1 python evaluation/complexity_analysis.py --model_name dada
   ```

3. **Monitor GPU Usage**:
   ```bash
   # In another terminal, monitor GPU
   watch -n 1 nvidia-smi
   ```

4. **Check Warnings**: The script now warns you:
   ```
   ⚠️  WARNING: Other processes detected on GPU!
   ⚠️  WARNING: GPU memory already in use!
   ```

### What to Do If You See Warnings

If the script warns about concurrent processes:

1. **For Accurate Memory Measurements**:
   - Stop other GPU processes
   - Run profiling in isolation
   - Use a dedicated GPU if available

2. **For Relative Comparisons**:
   - If all models run under the same conditions, relative comparisons may still be valid
   - But absolute numbers may be affected

3. **For Timing Measurements**:
   - More runs (increase `--num_runs`) can average out interference
   - Standard deviation will show variability

### Example Output

When you run the script, you'll now see:

```
============================================================
TimeRCD Computational Complexity Analysis
============================================================
Device: cuda
Model: timercd
...

GPU Isolation Check: GPU appears isolated

Initial GPU Memory State:
  Allocated: 0.00 MB
  Reserved: 0.00 MB
  Free: 79872.00 MB
  Total: 81920.00 MB

Initial System Memory: 1234.56 MB
```

If there are issues:
```
⚠️  WARNING: Other processes detected on GPU!
   PID 12345 (python): 1024MB
   PID 12346 (python): 2048MB
⚠️  WARNING: GPU memory already in use!
   Memory measurements may include allocations from other processes.
```

## Conclusion

The improved profiling code addresses several measurement issues:

1. ✅ More accurate timing with CUDA events and perf_counter
2. ✅ Comprehensive memory tracking (GPU + system)
3. ✅ Better statistics (std deviation)
4. ✅ Proper handling of CPU vs GPU models
5. ✅ Clear documentation of measurement differences

However, **fair comparisons still require**:
- Same device (all GPU or all CPU)
- Same operation type (inference-only vs training+inference)
- Clear documentation of what's being measured
- Multiple runs for statistical significance

When publishing results, consider:
- Separate tables for inference-only comparisons
- Notes about Chronos including training time
- System memory for models that use CPU
- Standard deviations for all measurements

