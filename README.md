# GPU Benchmark by [United Compute](https://www.unitedcompute.ai)

A simple CLI tool to benchmark your GPU's performance across various AI models (Stable Diffusion, LLMs, etc.) and compare results in our global benchmark results.

![United Compute Logo](https://www.unitedcompute.ai/logo.png)

## Installation

For Nvidia GPUs (CUDA)
```bash
pip install gpu-benchmark
```

For AMD GPUs (ROCm, currently Linux only)
```bash
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.3 gpu-benchmark.[rocm]
```

## Usage

Run the benchmark (takes 5 minutes after the pipeline is loaded):

```bash
gpu-benchmark
```

### Available Benchmarks

You can specify which model to benchmark using the `--model` flag:

**Stable Diffusion 1.5 (Default)**
```bash
gpu-benchmark --model stable-diffusion-1-5
```

**Qwen 3.0 6B (LLM Inference)**
```bash
gpu-benchmark --model qwen3-0-6b
```

**nanoGPT (LLM Training)**
```bash
gpu-benchmark --model nanogpt-train
```

### Optional Arguments

If you're running on a cloud provider, specify it with the `--provider` flag:

```bash
gpu-benchmark --provider runpod
```

For multi-GPU systems, you can select a specific GPU like this:

1. Using the `--gpu` flag:

```bash
gpu-benchmark --gpu 1  # Uses GPU index 1
```

The tool will:

1. Load the selected model (Stable Diffusion, Qwen, or nanoGPT)
2. Run the benchmark for 5 minutes
3. Track performance metrics (throughput/iterations) and GPU temperature
4. Upload results to the [United Compute Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)

## What it measures

- **Benchmark Score**: Number of iterations or images generated in 5 minutes (model-dependent)
- **GPU Model**: The specific model of your GPU (e.g., NVIDIA GeForce RTX 4090)
- **Max Heat**: Maximum GPU temperature reached (°C)
- **Avg Heat**: Average GPU temperature during the benchmark (°C)
- **Country**: Your location (detected automatically)
- **GPU Power**: Power consumption in watts (W)
- **GPU Memory**: Total GPU memory in gigabytes (GB)
- **Platform**: Operating system information
- **Acceleration**: CUDA/ROCM version
- **PyTorch Version**: PyTorch library version

## Requirements

- CUDA-compatible NVIDIA GPU or ROCm-compatible AMD GPU (Linux Only)
- Python 3.8+

## Links

- [Official Website](https://www.unitedcompute.ai)
- [GPU Benchmark Results](https://www.unitedcompute.ai/gpu-benchmark)
