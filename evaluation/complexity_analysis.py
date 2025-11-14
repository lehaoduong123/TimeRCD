"""
Computational Complexity Analysis Script for TimeRCD

This script analyzes the computational complexity (time and space) of TimeRCD
by measuring performance across different sequence lengths and configurations.
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.time_rcd.ts_encoder_bi_bias import TimeSeriesEncoder
from models.time_rcd.time_rcd_config import TimeRCDConfig, default_config
from models.time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


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
    peak_memory_mb: float
    model_params: int
    flops: Optional[int] = None


class ComplexityProfiler:
    """Profiler for analyzing computational complexity of TimeRCD."""
    
    def __init__(self, config: TimeRCDConfig, device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.hooks = []
        self.timings = {}
        
    def _setup_model(self):
        """Initialize the model for profiling."""
        if self.model is None:
            self.model = TimeSeriesPretrainModel(self.config).to(self.device)
            self.model.eval()
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
    
    def profile_forward_pass(self, 
                             seq_len: int, 
                             num_features: int, 
                             batch_size: int = 1,
                             num_runs: int = 10,
                             warmup_runs: int = 3) -> ComplexityMetrics:
        """
        Profile a single forward pass with given parameters.
        
        Args:
            seq_len: Sequence length
            num_features: Number of features
            batch_size: Batch size
            num_runs: Number of runs for averaging
            warmup_runs: Number of warmup runs
            
        Returns:
            ComplexityMetrics object with timing and memory information
        """
        self._setup_model()
        self.config.ts_config.num_features = num_features
        
        # Create dummy input data
        time_series = torch.randn(batch_size, seq_len, num_features, device=self.device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(time_series, attention_mask)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            initial_memory = 0
        
        # Timing measurements
        encoder_times = []
        attention_times = []
        recon_head_times = []
        anomaly_head_times = []
        total_times = []
        
        with torch.no_grad():
            for run in range(num_runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                
                # Forward through encoder
                encoder_start = time.time()
                local_embeddings = self.model.ts_encoder(time_series, attention_mask)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                encoder_time = time.time() - encoder_start
                encoder_times.append(encoder_time)
                
                # Forward through reconstruction head
                recon_start = time.time()
                _ = self.model.reconstruction_head(local_embeddings)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                recon_time = time.time() - recon_start
                recon_head_times.append(recon_time)
                
                # Forward through anomaly head
                anomaly_start = time.time()
                _ = self.model.anomaly_head(local_embeddings)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                anomaly_time = time.time() - anomaly_start
                anomaly_head_times.append(anomaly_time)
                
                # Total time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_time = time.time() - start_time
                total_times.append(total_time)
        
        # Measure peak memory
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            peak_memory = 0
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Estimate FLOPs
        flops = self.estimate_flops(seq_len, num_features, batch_size)
        
        # Calculate average times
        metrics = ComplexityMetrics(
            sequence_length=seq_len,
            num_features=num_features,
            batch_size=batch_size,
            total_time=np.mean(total_times),
            encoder_time=np.mean(encoder_times),
            attention_time=np.mean(encoder_times) * 0.8,  # Approximate attention time
            reconstruction_head_time=np.mean(recon_head_times),
            anomaly_head_time=np.mean(anomaly_head_times),
            peak_memory_mb=peak_memory,
            model_params=total_params,
            flops=flops
        )
        
        return metrics
    
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
                print(f"✓ (Time: {metrics.total_time*1000:.2f}ms, Memory: {metrics.peak_memory_mb:.2f}MB)")
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
        report["results"].append({
            "sequence_length": r.sequence_length,
            "num_features": r.num_features,
            "batch_size": r.batch_size,
            "total_time_ms": r.total_time * 1000,
            "encoder_time_ms": r.encoder_time * 1000,
            "reconstruction_head_time_ms": r.reconstruction_head_time * 1000,
            "anomaly_head_time_ms": r.anomaly_head_time * 1000,
            "peak_memory_mb": r.peak_memory_mb,
            "model_parameters": r.model_params,
            "time_per_element_ms": (r.total_time * 1000) / r.sequence_length,
            "memory_per_element_mb": r.peak_memory_mb / r.sequence_length,
            "flops": r.flops if r.flops else None,
            "flops_per_element": (r.flops / r.sequence_length) if r.flops else None
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
    parser.add_argument('--output_dir', type=str, default='evaluation/complexity_plots',
                       help='Output directory for plots and reports')
    
    args = parser.parse_args()
    
    # Load configuration
    config = default_config
    config.ts_config.num_features = args.num_features
    
    print("=" * 60)
    print("TimeRCD Computational Complexity Analysis")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Sequence lengths: {args.seq_lengths}")
    print(f"Features: {args.num_features}, Batch size: {args.batch_size}")
    print(f"Runs per config: {args.num_runs}")
    print("=" * 60)
    
    # Create profiler
    profiler = ComplexityProfiler(config, device=args.device)
    
    # Run analysis
    results = profiler.analyze_scaling(
        seq_lengths=args.seq_lengths,
        num_features=args.num_features,
        batch_size=args.batch_size,
        num_runs=args.num_runs
    )
    
    if not results:
        print("No results collected. Exiting.")
        return
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_complexity_analysis(results, save_dir=args.output_dir)
    
    # Generate report
    print("\nGenerating complexity report...")
    report_path = os.path.join(args.output_dir, 'complexity_report.json')
    generate_complexity_report(results, save_path=report_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total configurations tested: {len(results)}")
    print(f"\nTime Complexity:")
    print(f"  Min: {min(r.total_time*1000 for r in results):.2f} ms (seq_len={min(r.sequence_length for r in results)})")
    print(f"  Max: {max(r.total_time*1000 for r in results):.2f} ms (seq_len={max(r.sequence_length for r in results)})")
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


if __name__ == "__main__":
    main()

