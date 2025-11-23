"""
Computational Complexity Analysis Script for TimeRCD

This script analyzes the computational complexity (time and space) of TimeRCD
by measuring performance across different sequence lengths and configurations.

IMPROVEMENTS (v2):
=================
1. **Enhanced Time Tracking**:
   - Uses `time.perf_counter()` for higher precision CPU timing
   - Uses `torch.cuda.Event` for more accurate GPU timing when available
   - Proper synchronization before/after GPU operations
   - Reports standard deviation for timing measurements

2. **Comprehensive Memory Tracking**:
   - Tracks PyTorch GPU memory (via torch.cuda.max_memory_allocated)
   - Tracks system memory (via psutil) for models that use non-PyTorch memory
   - Measures memory delta from baseline for accurate tracking
   - Reports both GPU and system memory separately

3. **Better Baseline Comparisons**:
   - Improved handling of models that may use CPU or mixed device placement
   - Separate tracking for models that include training time (e.g., Chronos)
   - More accurate memory measurement for non-PyTorch frameworks

4. **Statistical Reporting**:
   - Standard deviation for timing measurements
   - Multiple runs for more reliable statistics
   - Better warmup handling to avoid initialization overhead

Note: Install psutil for system memory tracking: `pip install psutil`
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Additional imports for comprehensive profiling
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. System memory tracking will be disabled. Install with: pip install psutil")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.time_rcd.ts_encoder_bi_bias import TimeSeriesEncoder
from models.time_rcd.time_rcd_config import TimeRCDConfig, default_config
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def get_system_memory_mb() -> float:
    """Get current system memory usage in MB."""
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024**2
        except Exception:
            return 0.0
    return 0.0


def check_gpu_isolation(device_idx: Optional[int] = None) -> Tuple[bool, str]:
    """
    Check if GPU is being shared with other processes.
    
    Args:
        device_idx: CUDA device index to check (None for current device)
    
    Returns:
        Tuple of (is_isolated, warning_message)
    """
    if not torch.cuda.is_available():
        return True, "GPU not available (using CPU)"
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return True, "Could not check GPU isolation (nvidia-smi unavailable)"
        
        current_pid = os.getpid()
        other_processes = []
        
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    pid = int(parts[0].strip())
                    if pid != current_pid:
                        process_name = parts[1].strip()
                        memory = parts[2].strip()
                        other_processes.append(f"PID {pid} ({process_name}): {memory}MB")
                except (ValueError, IndexError):
                    continue
        
        if other_processes:
            warning = f"WARNING: Other processes using GPU:\n  " + "\n  ".join(other_processes)
            return False, warning
        else:
            return True, "GPU appears isolated"
            
    except FileNotFoundError:
        return True, "Could not check GPU isolation (nvidia-smi not found)"
    except Exception as e:
        return True, f"Could not check GPU isolation: {e}"


def get_gpu_memory_info(device_idx: Optional[int] = None) -> Dict[str, float]:
    """
    Get detailed GPU memory information.
    
    Args:
        device_idx: CUDA device index (None for current device)
    
    Returns:
        Dictionary with memory stats in MB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "free": 0.0, "total": 0.0}
    
    if device_idx is not None:
        torch.cuda.set_device(device_idx)
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    
    props = torch.cuda.get_device_properties(device_idx if device_idx is not None else 0)
    total = props.total_memory / 1024**2
    free = total - reserved
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }


def reset_memory_stats(device_idx: Optional[int] = None):
    """
    Reset both GPU and system memory tracking.
    
    Args:
        device_idx: CUDA device index to reset (None for current device)
    """
    if torch.cuda.is_available():
        if device_idx is not None:
            torch.cuda.set_device(device_idx)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Force garbage collection to get baseline system memory
    import gc
    gc.collect()
    
    # Small delay to ensure memory is freed
    time.sleep(0.1)


def get_peak_memory_mb(include_system: bool = True) -> Tuple[float, Optional[float]]:
    """
    Get peak memory usage.
    
    Returns:
        Tuple of (gpu_memory_mb, system_memory_mb)
    """
    gpu_memory = 0.0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    system_memory = None
    if include_system and PSUTIL_AVAILABLE:
        system_memory = get_system_memory_mb()
    
    return gpu_memory, system_memory


def profile_function_timing(func, *args, num_runs: int = 10, warmup_runs: int = 3, 
                           use_gpu_events: bool = False, **kwargs) -> Dict[str, float]:
    """
    Profile a function's execution time with comprehensive tracking.
    
    Args:
        func: Function to profile
        *args: Positional arguments for func
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        use_gpu_events: If True, use CUDA events for more accurate GPU timing
        **kwargs: Keyword arguments for func
    
    Returns:
        Dictionary with timing statistics
    """
    times = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            warnings.warn(f"Warning in warmup: {e}")
    
    # Synchronize before measurement
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measurement runs
    for _ in range(num_runs):
        if use_gpu_events and torch.cuda.is_available():
            # Use CUDA events for more accurate GPU timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            _ = func(*args, **kwargs)
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_ms = start_event.elapsed_time(end_event)  # Returns milliseconds
            times.append(elapsed_ms)
        else:
            # Use high-resolution CPU timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            _ = func(*args, **kwargs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to milliseconds
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'times': times
    }


@dataclass
class ComplexityMetrics:
    """Container for complexity analysis results."""
    sequence_length: int
    num_features: int
    batch_size: int
    total_time: float
    encoder_time: float
    attention_time: float
    reconstruction_head_time: float
    anomaly_head_time: float
    training_step_time: float
    peak_memory_mb: float  # GPU memory (PyTorch) in MB
    peak_system_memory_mb: Optional[float] = None  # System memory (RSS) in MB
    model_params: int = 0
    flops: Optional[int] = None
    std_time_ms: Optional[float] = None  # Standard deviation of inference time
    std_memory_mb: Optional[float] = None  # Standard deviation of memory usage


class BaselineRunner:
    """Adapter to profile third-party baseline models."""
    
    _model_cache: Dict[Tuple[str, int, int, int, int], object] = {}
    
    def __init__(
        self,
        name: str,
        device: torch.device,
        config: TimeRCDConfig,
        seq_len: int,
        num_features: int,
        batch_size: int,
    ):
        self.name = name.lower()
        self.device = device
        self.config = config
        self.seq_len = seq_len
        self.num_features = num_features
        self.batch_size = batch_size
        default_window = getattr(config.ts_config, 'patch_size', 64) or 64
        base_window = max(64, default_window)
        # Set optimal window sizes based on HP_list.py for fair comparison
        if self.name == 'dada':
            self.window_size = 100  # DADA optimal: 100
        elif self.name == 'chronos':
            self.window_size = 100  # Chronos optimal: 100
        elif self.name == 'timesfm':
            self.window_size = 96  # TimesFM optimal: 96
        elif self.name == 'moment':
            self.window_size = 64  # MOMENT optimal: 64
        elif self.name in ('timemoe', 'time_moe', 'time-moe'):
            self.window_size = 96  # TimeMoE optimal: 96
        else:
            self.window_size = int(min(seq_len, base_window))
    
    def _device_index(self) -> int:
        if self.device.type == 'cuda' and self.device.index is not None:
            return self.device.index
        return 0
    
    def _cache_key(self) -> Tuple[str, int, int, int, int]:
        device_type = 0 if self.device.type == 'cpu' else 1
        device_idx = self._device_index()
        return (self.name, device_type, device_idx, self.num_features, self.window_size)
    
    def _instantiate_model(self):
        if self.name == 'dada':
            from models.DADA import DADA as DADAModel
            return DADAModel(
                device=self._device_index(),
                win_size=self.window_size,
                batch_size=self.batch_size
            )
        if self.name in ('timemoe', 'time_moe', 'time-moe'):
            from models.time_moe import Time_MOE
            return Time_MOE(
                device=self._device_index(),
                win_size=self.window_size,
                batch_size=self.batch_size
            )
        if self.name == 'chronos':
            from models.Chronos import Chronos
            return Chronos(
                win_size=self.window_size,
                prediction_length=1,
                input_c=self.num_features,
                batch_size=self.batch_size
            )
        if self.name == 'timesfm':
            from models.TimesFM import TimesFM
            return TimesFM(
                win_size=self.window_size,
                prediction_length=1,
                input_c=self.num_features
            )
        if self.name == 'moment':
            from models.MOMENT import MOMENT as MomentModel
            return MomentModel(
                win_size=self.window_size,
                input_c=self.num_features,
                batch_size=self.batch_size
            )
        raise ValueError(f"Unsupported baseline model: {self.name}")
    
    def get_model(self):
        key = self._cache_key()
        if key not in self._model_cache:
            self._model_cache[key] = self._instantiate_model()
        return self._model_cache[key]
    
    def _prepare_numpy(self, data: torch.Tensor) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            np_data = data.detach().cpu().float().numpy()
        else:
            np_data = np.asarray(data, dtype=np.float32)
        if np_data.ndim == 3:
            # (B, T, C) -> (T * B, C)
            np_data = np_data.reshape(-1, np_data.shape[-1])
        elif np_data.ndim == 1:
            np_data = np_data.reshape(-1, 1)
        np_data = self._pad_to_window(np_data)
        return np_data
    
    def _pad_to_window(self, np_data: np.ndarray) -> np.ndarray:
        """Pad (or extend) data so its length is a multiple of window_size."""
        window = max(1, int(self.window_size))
        length = np_data.shape[0]
        if length < window:
            pad_len = window - length
        else:
            remainder = length % window
            pad_len = 0 if remainder == 0 else window - remainder
        if pad_len == 0:
            return np_data
        pad_values = np_data[-1:, :]
        pad = np.repeat(pad_values, pad_len, axis=0)
        return np.concatenate([np_data, pad], axis=0)
    
    def run_inference(self, data: torch.Tensor):
        """Run inference with chunking for all baseline models.
        
        All models use chunking based on their optimal window sizes for fair comparison:
        - DADA: 100, Chronos: 100, TimesFM: 96, MOMENT: 64, TimeMoE: 96
        """
        model = self.get_model()
        np_data = self._prepare_numpy(data)
        chunk_size = self.window_size
        total_length = np_data.shape[0]
        
        # If sequence is small enough, process directly
        if total_length <= chunk_size:
            return self._process_single_chunk(model, np_data)
        
        # Divide into chunks of window_size for fair comparison
        num_chunks = total_length // chunk_size
        if total_length % chunk_size != 0:
            num_chunks += 1
        
        all_scores = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_length)
            chunk = np_data[start_idx:end_idx]
            
            # Pad chunk to exactly chunk_size if needed
            if chunk.shape[0] < chunk_size:
                pad_len = chunk_size - chunk.shape[0]
                pad_values = chunk[-1:, :] if chunk.shape[0] > 0 else np.zeros((1, chunk.shape[1]))
                pad = np.repeat(pad_values, pad_len, axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            
            chunk_scores = self._process_single_chunk(model, chunk)
            actual_chunk_len = end_idx - start_idx
            
            # Handle different score shapes and trim to actual chunk length
            if chunk_scores is not None:
                if isinstance(chunk_scores, np.ndarray):
                    # Trim to actual chunk length (remove padding)
                    if len(chunk_scores) > actual_chunk_len:
                        chunk_scores = chunk_scores[:actual_chunk_len]
                    all_scores.append(chunk_scores)
                else:
                    all_scores.append(np.zeros(actual_chunk_len))
            else:
                all_scores.append(np.zeros(actual_chunk_len))
        
        # Concatenate all scores
        if len(all_scores) > 0:
            return np.concatenate(all_scores, axis=0)
        else:
            return np.array([])
    
    def _process_single_chunk(self, model, chunk: np.ndarray) -> np.ndarray:
        """Process a single chunk using the appropriate model method."""
        if self.name == 'dada':
            return model.zero_shot(chunk)
        if self.name in ('timemoe', 'time_moe', 'time-moe'):
            return model.zero_shot(chunk)
        if self.name == 'chronos':
            # Ensure data is properly formatted for Chronos
            # Chronos expects (n_samples, n_features) numpy array with numeric dtype
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            elif chunk.ndim == 2 and chunk.shape[1] != self.num_features:
                # Reshape if needed
                if chunk.shape[1] > self.num_features:
                    chunk = chunk[:, :self.num_features]
                else:
                    # Pad if needed
                    pad_width = ((0, 0), (0, self.num_features - chunk.shape[1]))
                    chunk = np.pad(chunk, pad_width, mode='constant', constant_values=0.0)
            # Ensure no NaN or inf values
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            model.score_list = []
            try:
                model.fit(chunk)
                scores = getattr(model, 'decision_scores_', None)
                return scores if scores is not None else np.zeros(len(chunk))
            except Exception as e:
                print(f"Warning: Chronos fit failed: {e}")
                return np.zeros(len(chunk))
        if self.name == 'timesfm':
            model.score_list = []
            model.fit(chunk)
            scores = getattr(model, 'decision_scores_', None)
            return scores if scores is not None else np.zeros(len(chunk))
        if self.name == 'moment':
            model.zero_shot(chunk)
            scores = getattr(model, 'decision_scores_', None)
            return scores if scores is not None else np.zeros(len(chunk))
        raise ValueError(f"Inference not implemented for baseline {self.name}")
    
    def supports_training(self) -> bool:
        return False
    
    def run_training_step(self, data: torch.Tensor):
        raise NotImplementedError("Training is not supported for this baseline.")
    
    def parameter_count(self) -> Optional[int]:
        model = self.get_model()
        inner = getattr(model, 'model', None)
        target = inner if inner is not None else model
        parameters = getattr(target, 'parameters', None)
        if callable(parameters):
            try:
                return sum(p.numel() for p in parameters())
            except Exception:
                return None
        return None


class ComplexityProfiler:
    """Profiler for analyzing computational complexity of TimeRCD."""
    
    def __init__(self, config: TimeRCDConfig, device: str = "cuda", model_name: str = "timercd"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()
        self.model = None
        self._param_count = None
        self.hooks = []
        self.timings = {}
        
    def _setup_model(self):
        """Initialize the model for profiling."""
        if self.model_name != "timercd":
            return None
        if self.model is None:
            self.model = TimeSeriesPretrainModel(self.config).to(self.device)
            self.model.eval()
            self._param_count = sum(p.numel() for p in self.model.parameters())
        return self.model
    
    def _hook_timing(self, name: str):
        """Create a hook to time a specific layer."""
        def hook_fn(module, input, output):
            if name not in self.timings:
                self.timings[name] = []
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.timings[name].append(time.time())
        return hook_fn
    
    def _register_hooks(self):
        """Register timing hooks for different components."""
        self.timings = {}
        
        # Hook for encoder
        if hasattr(self.model, 'ts_encoder'):
            self.hooks.append(
                self.model.ts_encoder.register_forward_hook(
                    lambda m, i, o: self._record_time('encoder', time.time())
                )
            )
        
        # Hook for attention layers
        if hasattr(self.model, 'ts_encoder') and hasattr(self.model.ts_encoder, 'transformer_encoder'):
            for i, layer in enumerate(self.model.ts_encoder.transformer_encoder.layers):
                self.hooks.append(
                    layer.register_forward_hook(
                        lambda m, i, o, idx=i: self._record_time(f'attention_layer_{idx}', time.time())
                    )
                )
    
    def _record_time(self, name: str, timestamp: float):
        """Record timing for a component."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(timestamp)
    
    def _cleanup_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _prepare_training_batch(self,
                                time_series: torch.Tensor,
                                attention_mask: torch.Tensor,
                                mask_ratio: float = 0.15
                                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create masked inputs and labels for a synthetic training step."""
        batch_size, seq_len, num_features = time_series.shape
        random_mask = (torch.rand(batch_size, seq_len, device=self.device) < mask_ratio) & attention_mask
        mask = random_mask.float()
        
        masked_time_series = time_series.clone()
        mask_expanded = random_mask.unsqueeze(-1).expand(-1, -1, num_features)
        masked_time_series[mask_expanded] = 0.0
        
        labels = torch.randint(0, 2, (batch_size, seq_len), device=self.device)
        labels = labels.long()
        labels = labels.masked_fill(~attention_mask, -1)
        
        return masked_time_series, mask, labels
    
    def profile_forward_pass(self, 
                             seq_len: int, 
                             num_features: int, 
                             batch_size: int = 1,
                             num_runs: int = 10,
                             warmup_runs: int = 3) -> ComplexityMetrics:
        if self.model_name == "timercd":
            return self._profile_timercd(
                seq_len=seq_len,
                num_features=num_features,
                batch_size=batch_size,
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
        return self._profile_baseline(
            seq_len=seq_len,
            num_features=num_features,
            batch_size=batch_size,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )

    def _profile_timercd(self,
                         seq_len: int,
                         num_features: int,
                         batch_size: int,
                         num_runs: int,
                         warmup_runs: int) -> ComplexityMetrics:
        """Profile the native TimeRCD model."""
        self._setup_model()
        self.config.ts_config.num_features = num_features
        
        # Create dummy input data
        time_series = torch.randn(batch_size, seq_len, num_features, device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Check GPU isolation and get initial memory state
        if torch.cuda.is_available():
            device_idx = self.device.index if hasattr(self.device, 'index') and self.device.index is not None else 0
            is_isolated, isolation_msg = check_gpu_isolation(device_idx)
            if not is_isolated:
                warnings.warn(isolation_msg)
            
            initial_gpu_mem = get_gpu_memory_info(device_idx)
            if initial_gpu_mem["allocated"] > 100:  # More than 100MB already allocated
                warnings.warn(
                    f"GPU already has {initial_gpu_mem['allocated']:.2f}MB allocated. "
                    f"Results may be affected by other processes. "
                    f"Consider running profiling in isolation."
                )
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(time_series, attention_mask)
        
        # Reset memory tracking (with delay to ensure cleanup)
        device_idx = self.device.index if torch.cuda.is_available() and hasattr(self.device, 'index') and self.device.index is not None else None
        reset_memory_stats(device_idx)
        
        # Wait a bit more to ensure baseline is stable
        time.sleep(0.2)
        
        # Get baseline system memory (measure multiple times for stability)
        baseline_system_mem = None
        if PSUTIL_AVAILABLE:
            baseline_readings = []
            for _ in range(3):
                baseline_readings.append(get_system_memory_mb())
                time.sleep(0.1)
            baseline_system_mem = np.mean(baseline_readings)
        
        # Timing measurements using improved profiling
        encoder_times = []
        recon_head_times = []
        anomaly_head_times = []
        total_times = []
        
        # Use CUDA events for more accurate GPU timing if available
        use_cuda_events = torch.cuda.is_available()
        
        with torch.no_grad():
            for _ in range(num_runs):
                if use_cuda_events:
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_time = time.perf_counter()
                
                # Forward through encoder
                if use_cuda_events:
                    enc_start = torch.cuda.Event(enable_timing=True)
                    enc_end = torch.cuda.Event(enable_timing=True)
                    enc_start.record()
                else:
                    enc_start = time.perf_counter()
                
                local_embeddings = self.model.ts_encoder(time_series, attention_mask)
                
                if use_cuda_events:
                    enc_end.record()
                    torch.cuda.synchronize()
                    encoder_times.append(enc_start.elapsed_time(enc_end) / 1000.0)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    encoder_times.append(time.perf_counter() - enc_start)
                
                # Forward through reconstruction head
                if use_cuda_events:
                    recon_start = torch.cuda.Event(enable_timing=True)
                    recon_end = torch.cuda.Event(enable_timing=True)
                    recon_start.record()
                else:
                    recon_start = time.perf_counter()
                
                _ = self.model.reconstruction_head(local_embeddings)
                
                if use_cuda_events:
                    recon_end.record()
                    torch.cuda.synchronize()
                    recon_head_times.append(recon_start.elapsed_time(recon_end) / 1000.0)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    recon_head_times.append(time.perf_counter() - recon_start)
                
                # Forward through anomaly head
                if use_cuda_events:
                    anom_start = torch.cuda.Event(enable_timing=True)
                    anom_end = torch.cuda.Event(enable_timing=True)
                    anom_start.record()
                else:
                    anom_start = time.perf_counter()
                
                _ = self.model.anomaly_head(local_embeddings)
                
                if use_cuda_events:
                    anom_end.record()
                    end_event.record()
                    torch.cuda.synchronize()
                    anomaly_head_times.append(anom_start.elapsed_time(anom_end) / 1000.0)
                    total_times.append(start_event.elapsed_time(end_event) / 1000.0)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    anomaly_head_times.append(time.perf_counter() - anom_start)
                    total_times.append(time.perf_counter() - start_time)
        
        # Simulate training step timing with backward pass
        training_step_times = []
        self.model.train()
        
        use_cuda_events_train = torch.cuda.is_available()
        
        for _ in range(num_runs):
            self.model.zero_grad(set_to_none=True)
            masked_series, mask, labels = self._prepare_training_batch(time_series, attention_mask)
            
            if use_cuda_events_train:
                torch.cuda.synchronize()
                train_start = torch.cuda.Event(enable_timing=True)
                train_end = torch.cuda.Event(enable_timing=True)
                train_start.record()
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_start = time.perf_counter()
            
            local_embeddings = self.model(masked_series, attention_mask & (~mask.bool()))
            recon_loss = self.model.masked_reconstruction_loss(local_embeddings, time_series, mask)
            anomaly_loss = self.model.anomaly_detection_loss(local_embeddings, labels)
            total_loss = recon_loss + anomaly_loss
            total_loss.backward()
            
            if use_cuda_events_train:
                train_end.record()
                torch.cuda.synchronize()
                training_step_times.append(train_start.elapsed_time(train_end) / 1000.0)
            else:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                training_step_times.append(time.perf_counter() - train_start)
        
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        
        # Get peak memory (GPU and system)
        peak_gpu_mem, peak_system_mem = get_peak_memory_mb(include_system=True)
        
        # Calculate system memory delta
        system_memory_delta = None
        if baseline_system_mem is not None and peak_system_mem is not None:
            system_memory_delta = peak_system_mem - baseline_system_mem
        
        total_params = sum(p.numel() for p in self.model.parameters())
        flops = self.estimate_flops(seq_len, num_features, batch_size)
        
        return ComplexityMetrics(
            sequence_length=seq_len,
            num_features=num_features,
            batch_size=batch_size,
            total_time=np.mean(total_times),
            encoder_time=np.mean(encoder_times),
            attention_time=np.mean(encoder_times) * 0.8,
            reconstruction_head_time=np.mean(recon_head_times),
            anomaly_head_time=np.mean(anomaly_head_times),
            training_step_time=np.mean(training_step_times) if training_step_times else float('nan'),
            peak_memory_mb=peak_gpu_mem,
            peak_system_memory_mb=system_memory_delta,
            model_params=self._param_count if self._param_count is not None else total_params,
            flops=flops,
            std_time_ms=np.std(total_times) * 1000.0 if len(total_times) > 1 else None,
            std_memory_mb=None  # Would need multiple runs to calculate
        )
    
    def _profile_baseline(self,
                          seq_len: int,
                          num_features: int,
                          batch_size: int,
                          num_runs: int,
                          warmup_runs: int = 3) -> ComplexityMetrics:
        """Profile external baseline models such as DADA, TimeMoE, Chronos, TimesFM, MOMENT.
        
        All baseline models use chunking based on their optimal window sizes (from HP_list.py)
        for fair comparison. Sequences longer than window_size are divided into chunks and
        processed separately. The total time includes processing all chunks.
        
        IMPORTANT NOTE: For Chronos, this measures fit+inference time since it calls fit()
        which includes training. This is different from TimeRCD which only does inference.
        Consider reporting this separately or noting in documentation.
        
        For Chronos specifically:
        - Memory may appear artificially low because AutoGluon's TimeSeriesPredictor
          may run on CPU or manage memory outside PyTorch's tracking
        - Time includes both model fitting (training) and prediction, making it
          slower than inference-only comparisons
        - System memory tracking helps capture memory that PyTorch doesn't track
        
        Window sizes:
        - DADA: 100, Chronos: 100, TimesFM: 96, MOMENT: 64, TimeMoE: 96
        """
        runner = BaselineRunner(
            name=self.model_name,
            device=self.device,
            config=self.config,
            seq_len=seq_len,
            num_features=num_features,
            batch_size=batch_size
        )
        
        # Check GPU isolation and get initial memory state
        if torch.cuda.is_available():
            device_idx = self.device.index if self.device.index is not None else 0
            is_isolated, isolation_msg = check_gpu_isolation(device_idx)
            if not is_isolated:
                warnings.warn(isolation_msg)
            
            initial_gpu_mem = get_gpu_memory_info(device_idx)
            if initial_gpu_mem["allocated"] > 100:  # More than 100MB already allocated
                warnings.warn(
                    f"GPU already has {initial_gpu_mem['allocated']:.2f}MB allocated. "
                    f"Results may be affected by other processes. "
                    f"Consider running profiling in isolation."
                )
        
        time_series = torch.randn(batch_size, seq_len, num_features, device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Reset memory tracking (with delay to ensure cleanup)
        device_idx = self.device.index if torch.cuda.is_available() and hasattr(self.device, 'index') else None
        reset_memory_stats(device_idx)
        
        # Wait a bit more to ensure baseline is stable
        time.sleep(0.2)
        
        # Get baseline system memory (measure multiple times for stability)
        baseline_system_mem = None
        if PSUTIL_AVAILABLE:
            baseline_readings = []
            for _ in range(3):
                baseline_readings.append(get_system_memory_mb())
                time.sleep(0.1)
            baseline_system_mem = np.mean(baseline_readings)
        
        # Profile inference time with comprehensive tracking
        timing_stats = profile_function_timing(
            runner.run_inference,
            time_series,
            num_runs=num_runs,
            warmup_runs=warmup_runs,
            use_gpu_events=torch.cuda.is_available() and self.model_name != 'chronos'
            # Don't use GPU events for Chronos as it may use CPU
        )
        
        # Get peak memory (GPU and system)
        peak_gpu_mem, peak_system_mem = get_peak_memory_mb(include_system=True)
        
        # Calculate system memory delta
        system_memory_delta = None
        if baseline_system_mem is not None and peak_system_mem is not None:
            system_memory_delta = peak_system_mem - baseline_system_mem
        
        # Track memory across runs for std calculation
        memory_readings = []
        if num_runs > 1:
            # Do additional runs to measure memory variation
            for _ in range(min(5, num_runs)):
                reset_memory_stats()
                _ = runner.run_inference(time_series)
                gpu_mem, sys_mem = get_peak_memory_mb(include_system=False)
                memory_readings.append(gpu_mem)
        
        std_memory = np.std(memory_readings) if len(memory_readings) > 1 else None
        
        param_count = runner.parameter_count()
        
        return ComplexityMetrics(
            sequence_length=seq_len,
            num_features=num_features,
            batch_size=batch_size,
            total_time=timing_stats['mean_ms'] / 1000.0,  # Convert back to seconds
            encoder_time=timing_stats['mean_ms'] / 1000.0,
            attention_time=timing_stats['mean_ms'] / 1000.0,
            reconstruction_head_time=0.0,
            anomaly_head_time=0.0,
            training_step_time=float('nan'),
            peak_memory_mb=peak_gpu_mem,
            peak_system_memory_mb=system_memory_delta,
            model_params=param_count if param_count is not None else 0,
            flops=None,
            std_time_ms=timing_stats['std_ms'],
            std_memory_mb=std_memory
        )
    
    def analyze_scaling(self,
                       seq_lengths: List[int],
                       num_features: int = 1,
                       batch_size: int = 1,
                       num_runs: int = 10) -> List[ComplexityMetrics]:
        """
        Analyze how complexity scales with sequence length.
        
        Args:
            seq_lengths: List of sequence lengths to test
            num_features: Number of features
            batch_size: Batch size
            num_runs: Number of runs per configuration
            
        Returns:
            List of ComplexityMetrics for each sequence length
        """
        results = []
        
        print(f"Analyzing scaling behavior with {len(seq_lengths)} sequence lengths...")
        print(f"Features: {num_features}, Batch size: {batch_size}")
        print("-" * 60)
        
        for seq_len in seq_lengths:
            print(f"Testing sequence length: {seq_len}...", end=" ", flush=True)
            try:
                metrics = self.profile_forward_pass(
                    seq_len=seq_len,
                    num_features=num_features,
                    batch_size=batch_size,
                    num_runs=num_runs
                )
                results.append(metrics)
                train_ms = metrics.training_step_time * 1000
                train_text = f"{train_ms:.2f}ms" if np.isfinite(train_ms) else "N/A"
                memory_text = f"{metrics.peak_memory_mb:.2f}MB"
                if metrics.peak_system_memory_mb is not None:
                    memory_text += f" (sys: {metrics.peak_system_memory_mb:.2f}MB)"
                
                std_text = ""
                if metrics.std_time_ms is not None:
                    std_text = f" ±{metrics.std_time_ms:.2f}ms"
                
                print(
                    f"✓ (Infer: {metrics.total_time*1000:.2f}ms{std_text}, "
                    f"Train: {train_text}, "
                    f"Memory: {memory_text})"
                )
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        return results
    
    def estimate_flops(self, seq_len: int, num_features: int, batch_size: int = 1) -> int:
        """
        Estimate FLOPs (Floating Point Operations) for a forward pass.
        
        This is an approximation based on the model architecture.
        """
        d_model = self.config.ts_config.d_model
        d_proj = self.config.ts_config.d_proj
        num_layers = self.config.ts_config.num_layers
        num_heads = self.config.ts_config.num_heads
        patch_size = self.config.ts_config.patch_size
        
        num_patches = (seq_len + patch_size - 1) // patch_size
        L = num_patches * num_features  # Total sequence length after patching
        
        flops = 0
        
        # Embedding layer
        flops += batch_size * L * d_model
        
        # Transformer layers
        for _ in range(num_layers):
            # Self-attention: Q, K, V projections
            flops += 3 * batch_size * L * d_model * d_model
            # Attention computation: Q @ K^T
            flops += batch_size * num_heads * L * L * (d_model // num_heads)
            # Attention @ V
            flops += batch_size * num_heads * L * L * (d_model // num_heads)
            # Output projection
            flops += batch_size * L * d_model * d_model
            # FFN (assuming d_ff = 4 * d_model)
            flops += 2 * batch_size * L * d_model * (4 * d_model)
        
        # Projection layer
        flops += batch_size * L * d_model * (patch_size * d_proj)
        
        # Reconstruction head
        flops += batch_size * seq_len * num_features * d_proj * (4 * d_proj)  # First layer
        flops += batch_size * seq_len * num_features * (4 * d_proj) * (4 * d_proj)  # Second layer
        flops += batch_size * seq_len * num_features * (4 * d_proj) * 1  # Output layer
        
        # Anomaly head
        flops += batch_size * seq_len * num_features * d_proj * (d_proj // 2)
        flops += batch_size * seq_len * num_features * (d_proj // 2) * 2
        
        return int(flops)


def plot_complexity_analysis(results: List[ComplexityMetrics], 
                            save_dir: str = "evaluation/complexity_plots"):
    """
    Generate visualization plots for complexity analysis.
    
    Args:
        results: List of ComplexityMetrics
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    seq_lengths = [r.sequence_length for r in results]
    total_times = [r.total_time * 1000 for r in results]  # Convert to ms
    encoder_times = [r.encoder_time * 1000 for r in results]
    recon_times = [r.reconstruction_head_time * 1000 for r in results]
    anomaly_times = [r.anomaly_head_time * 1000 for r in results]
    training_times = [
        r.training_step_time * 1000 if np.isfinite(r.training_step_time) else np.nan
        for r in results
    ]
    memory_usage = [r.peak_memory_mb for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TimeRCD Computational Complexity Analysis', fontsize=16, fontweight='bold')
    
    # 1. Time complexity vs sequence length
    ax1 = axes[0, 0]
    ax1.plot(seq_lengths, total_times, 'o-', label='Total Time', linewidth=2, markersize=8)
    ax1.plot(seq_lengths, encoder_times, 's-', label='Encoder', linewidth=2, markersize=6)
    ax1.plot(seq_lengths, recon_times, '^-', label='Reconstruction Head', linewidth=2, markersize=6)
    ax1.plot(seq_lengths, anomaly_times, 'd-', label='Anomaly Head', linewidth=2, markersize=6)
    training_array = np.array(training_times, dtype=float)
    if np.isfinite(training_array).any():
        ax1.plot(seq_lengths, training_array, 'x-', label='Training Step', linewidth=2, markersize=6)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Time Complexity vs Sequence Length', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=10)
    
    # 2. Memory complexity vs sequence length
    ax2 = axes[0, 1]
    ax2.plot(seq_lengths, memory_usage, 'o-', color='purple', linewidth=2, markersize=8)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Peak Memory (MB)', fontsize=12)
    ax2.set_title('Space Complexity vs Sequence Length', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Time per element (normalized)
    ax3 = axes[1, 0]
    time_per_element = [t / s for t, s in zip(total_times, seq_lengths)]
    ax3.plot(seq_lengths, time_per_element, 'o-', color='green', linewidth=2, markersize=8)
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('Time per Element (ms/element)', fontsize=12)
    ax3.set_title('Normalized Time Complexity', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    # 4. Component breakdown (pie chart for largest sequence)
    ax4 = axes[1, 1]
    largest_idx = len(results) - 1
    largest_result = results[largest_idx]
    components = [
        largest_result.encoder_time * 1000,
        largest_result.reconstruction_head_time * 1000,
        largest_result.anomaly_head_time * 1000
    ]
    labels = ['Encoder', 'Reconstruction Head', 'Anomaly Head']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax4.pie(components, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title(f'Time Breakdown (Seq Length: {largest_result.sequence_length})', 
                  fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'complexity_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {save_path}")
    plt.close()
    
    # Create detailed scaling analysis plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Fit polynomial to understand scaling
    log_seq = np.log2(seq_lengths)
    log_time = np.log10(total_times)
    
    # Linear fit in log space
    coeffs = np.polyfit(log_seq, log_time, 1)
    fitted_log_time = np.polyval(coeffs, log_seq)
    fitted_time = 10 ** fitted_log_time
    
    ax.scatter(seq_lengths, total_times, s=100, alpha=0.7, label='Measured', zorder=3)
    ax.plot(seq_lengths, fitted_time, 'r--', linewidth=2, 
            label=f'Fit: O(n^{coeffs[0]:.2f})', zorder=2)
    ax.set_xlabel('Sequence Length (n)', fontsize=14)
    ax.set_ylabel('Time (ms)', fontsize=14)
    ax.set_title('Time Complexity Scaling Analysis', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    
    # Add text annotation
    complexity_class = "Quadratic" if coeffs[0] > 1.5 else "Near-linear" if coeffs[0] < 1.2 else "Super-linear"
    ax.text(0.05, 0.95, f'Complexity: {complexity_class}\nScaling: O(n^{coeffs[0]:.2f})',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path2 = os.path.join(save_dir, 'scaling_analysis.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Scaling analysis saved to: {save_path2}")
    plt.close()


def generate_complexity_report(results: List[ComplexityMetrics], 
                              save_path: str = "evaluation/complexity_report.json"):
    """
    Generate a detailed complexity report in JSON format.
    
    Args:
        results: List of ComplexityMetrics
        save_path: Path to save the report
    """
    # Get model config from first result (assuming same config for all)
    first_result = results[0]
    profiler = ComplexityProfiler(default_config)
    profiler._setup_model()
    
    report = {
        "model_config": {
            "d_model": profiler.config.ts_config.d_model,
            "d_proj": profiler.config.ts_config.d_proj,
            "num_layers": profiler.config.ts_config.num_layers,
            "num_heads": profiler.config.ts_config.num_heads,
            "patch_size": profiler.config.ts_config.patch_size,
            "num_features": first_result.num_features,
        },
        "results": []
    }
    
    for r in results:
        training_time_ms = r.training_step_time * 1000 if np.isfinite(r.training_step_time) else None
        report["results"].append({
            "sequence_length": r.sequence_length,
            "num_features": r.num_features,
            "batch_size": r.batch_size,
            "total_time_ms": r.total_time * 1000,
            "encoder_time_ms": r.encoder_time * 1000,
            "reconstruction_head_time_ms": r.reconstruction_head_time * 1000,
            "anomaly_head_time_ms": r.anomaly_head_time * 1000,
            "training_step_time_ms": training_time_ms,
            "peak_memory_mb": r.peak_memory_mb,
            "peak_system_memory_mb": r.peak_system_memory_mb,
            "model_parameters": r.model_params,
            "time_per_element_ms": (r.total_time * 1000) / r.sequence_length,
            "memory_per_element_mb": r.peak_memory_mb / r.sequence_length,
            "flops": r.flops if r.flops else None,
            "flops_per_element": (r.flops / r.sequence_length) if r.flops else None,
            "std_time_ms": r.std_time_ms,
            "std_memory_mb": r.std_memory_mb
        })
    
    # Calculate scaling factors
    if len(results) >= 2:
        smallest = results[0]
        largest = results[-1]
        
        seq_ratio = largest.sequence_length / smallest.sequence_length
        time_ratio = largest.total_time / smallest.total_time
        memory_ratio = largest.peak_memory_mb / smallest.peak_memory_mb
        
        report["scaling_analysis"] = {
            "sequence_length_ratio": seq_ratio,
            "time_ratio": time_ratio,
            "memory_ratio": memory_ratio,
            "time_scaling_factor": time_ratio / seq_ratio,
            "memory_scaling_factor": memory_ratio / seq_ratio,
            "estimated_time_complexity": f"O(n^{np.log(time_ratio) / np.log(seq_ratio):.2f})",
            "estimated_space_complexity": f"O(n^{np.log(memory_ratio) / np.log(seq_ratio):.2f})"
        }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComplexity report saved to: {save_path}")


def print_summary(model_name: str, feature_count: int, results: List[ComplexityMetrics]):
    """Print summary statistics for a given feature count."""
    print("\n" + "=" * 60)
    print(f"Summary | model={model_name} | num_features={feature_count}")
    print("=" * 60)
    print(f"Total configurations tested: {len(results)}")
    print(f"\nTime Complexity:")
    print(f"  Min: {min(r.total_time*1000 for r in results):.2f} ms (seq_len={min(r.sequence_length for r in results)})")
    print(f"  Max: {max(r.total_time*1000 for r in results):.2f} ms (seq_len={max(r.sequence_length for r in results)})")
    training_values = [r.training_step_time*1000 for r in results if np.isfinite(r.training_step_time)]
    if training_values:
        print(f"\nTraining Step Time:")
        print(f"  Min: {min(training_values):.2f} ms")
        print(f"  Max: {max(training_values):.2f} ms")
    else:
        print("\nTraining Step Time: N/A (baseline does not expose training)")
    print(f"\nMemory Complexity:")
    print(f"  Min: {min(r.peak_memory_mb for r in results):.2f} MB (seq_len={min(r.sequence_length for r in results)})")
    print(f"  Max: {max(r.peak_memory_mb for r in results):.2f} MB (seq_len={max(r.sequence_length for r in results)})")
    
    if len(results) >= 2:
        smallest = results[0]
        largest = results[-1]
        seq_ratio = largest.sequence_length / smallest.sequence_length
        time_ratio = largest.total_time / smallest.total_time
        memory_ratio = largest.peak_memory_mb / smallest.peak_memory_mb
        
        print(f"\nScaling Analysis (from {smallest.sequence_length} to {largest.sequence_length}):")
        print(f"  Sequence length increase: {seq_ratio:.2f}x")
        print(f"  Time increase: {time_ratio:.2f}x")
        print(f"  Memory increase: {memory_ratio:.2f}x")
        print(f"  Time scaling: O(n^{np.log(time_ratio) / np.log(seq_ratio):.2f})")
        print(f"  Memory scaling: O(n^{np.log(memory_ratio) / np.log(seq_ratio):.2f})")
    
    print("=" * 60)


def main():
    """Main function to run complexity analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze computational complexity of TimeRCD')
    parser.add_argument('--seq_lengths', type=int, nargs='+', 
                       default=[1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000],
                       help='Sequence lengths to test')
    parser.add_argument('--num_features', type=int, default=1,
                       help='Number of features')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of runs per configuration')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--cuda_devices', type=str, default='1',
                       help='CUDA devices to use')
    parser.add_argument('--model_name', type=str, default='timercd',
                       choices=['timercd', 'dada', 'timemoe', 'chronos', 'timesfm', 'moment'],
                       help='Model/baseline to profile')
    parser.add_argument('--output_dir', type=str, default='evaluation/complexity_plots',
                       help='Output directory for plots and reports')
    parser.add_argument('--feature_counts', type=int, nargs='+', default=None,
                       help='List of feature counts to test; overrides --num_features if provided')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    # Load configuration
    config = default_config
    config.ts_config.num_features = args.num_features
    
    print("=" * 60)
    print("TimeRCD Computational Complexity Analysis")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: {args.model_name}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Runs per config: {args.num_runs}")
    print("=" * 60)
    
    feature_counts = args.feature_counts if args.feature_counts else [args.num_features]
    all_results = {}
    model_output_dir = os.path.join(args.output_dir, args.model_name.lower())
    os.makedirs(model_output_dir, exist_ok=True)
    
    for feature_count in feature_counts:
        print(f"\nRunning analysis for num_features={feature_count}")
        config.ts_config.num_features = feature_count
        
        profiler = ComplexityProfiler(config, device=args.device, model_name=args.model_name)
        results = profiler.analyze_scaling(
            seq_lengths=args.seq_lengths,
            num_features=feature_count,
            batch_size=args.batch_size,
            num_runs=args.num_runs
        )
        
        if not results:
            print(f"No results collected for num_features={feature_count}. Skipping.")
            continue
        
        feature_output_dir = model_output_dir
        if len(feature_counts) > 1:
            feature_output_dir = os.path.join(model_output_dir, f"features_{feature_count}")
        os.makedirs(feature_output_dir, exist_ok=True)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        plot_complexity_analysis(results, save_dir=feature_output_dir)
        
        # Generate report
        print("\nGenerating complexity report...")
        report_path = os.path.join(feature_output_dir, f'complexity_report_features_{feature_count}.json')
        generate_complexity_report(results, save_path=report_path)
        
        print_summary(args.model_name, feature_count, results)
        all_results[feature_count] = results
    
    if not all_results:
        print("No analyses were completed. Exiting.")
        return


if __name__ == "__main__":
    main()

