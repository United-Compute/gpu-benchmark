# src/gpu_benchmark/main.py
from .benchmarks import stable_diffusion_1_5, qwen3_0_6b, nanogpt_train
from .database import upload_benchmark_results
import argparse
import torch 

# Import benchmark runners dynamically or add specific imports here later
# For now, let's assume functions like run_stable_diffusion_benchmark, run_llm_benchmark
# will be available from src.gpu_benchmark.benchmarks
# from .benchmarks import stable_diffusion # This will be created
# from .utils import get_clean_platform # This will be created, assuming get_clean_platform moves to utils

def main():
    """Entry point for the GPU benchmark command-line tool."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="GPU Benchmark by United Compute")
    parser.add_argument("--provider", type=str, help="Cloud provider (e.g., RunPod, AWS, GCP) or Private", default="Private")
    parser.add_argument("--gpu", type=int, help="GPU device index to use (defaults to CUDA_VISIBLE_DEVICES or 0)", default=None)
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model to benchmark (e.g., stable-diffusion-1-5, qwen3-0-6b, nanogpt-train)", 
        default="stable-diffusion-1-5",
        choices=["stable-diffusion-1-5", "qwen3-0-6b", "nanogpt-train"]
    )
    args = parser.parse_args()
    
    # If GPU device is specified, set it
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    # Convert provider to lowercase
    provider = args.provider.lower()
    
    # Simple start message
    print(f"GPU Benchmark starting for model: {args.model}...")
    print("This benchmark will run for 5 minutes")
    
    # Fixed duration
    duration = 300  # 300 seconds
    
    results = None
    if args.model == "stable-diffusion-1-5":
        print("Loading Stable Diffusion 1.5 pipeline...")
        pipe = stable_diffusion_1_5.load_pipeline() 
        print("Pipeline loaded successfully!")
        
        print("Running Stable Diffusion 1.5 benchmark...")
        results = stable_diffusion_1_5.run_benchmark(pipe=pipe, duration=duration)
    elif args.model == "qwen3-0-6b":
        print("Loading Qwen3-0-6B model...")
        model, tokenizer = qwen3_0_6b.load_pipeline()
        
        print("Running Qwen3-0-6B benchmark...")
        results = qwen3_0_6b.run_benchmark(model=model, tokenizer=tokenizer, duration=duration)
    elif args.model == "nanogpt-train":
        print("Loading nanoGPT for training benchmark...")
        pipeline = nanogpt_train.load_pipeline()
        print("Model loaded successfully!")
        
        print("Running nanoGPT training benchmark...")
        results = nanogpt_train.run_benchmark(pipeline=pipeline, duration=duration)
    else:
        print(f"Error: Model {args.model} not supported.")
        return

    # Only proceed if the benchmark completed successfully (not canceled)
    if results and results.get("completed", False):
        primary_metric_val = None
        max_temp_val = None
        avg_temp_val = None
        gpu_memory_val = None

        # Get the primary metric using the generic 'result' key
        primary_metric_val = results.get('result')

        if args.model == "stable-diffusion-1-5":
            max_temp_val = results.get('max_temp')
            avg_temp_val = results.get('avg_temp')
            gpu_memory_val = results.get('gpu_memory_total')
        elif args.model == "qwen3-0-6b":
            max_temp_val = results.get('max_temp')
            avg_temp_val = results.get('avg_temp')
            gpu_memory_val = results.get('gpu_memory_total')
        elif args.model == "nanogpt-train":
            max_temp_val = results.get('max_temp')
            avg_temp_val = results.get('avg_temp')
            gpu_memory_val = results.get('gpu_memory_total')
        
        # Build extra kwargs for training benchmarks
        extra_kwargs = {
            'gpu_power_watts': results.get('gpu_power_watts'),
            'gpu_memory_total': gpu_memory_val,
            'platform': results.get('platform'),
            'acceleration': results.get('acceleration'),
            'torch_version': results.get('torch_version')
        }
        
        # Add training-specific metrics
        if args.model == "nanogpt-train":
            extra_kwargs['iterations'] = results.get('iterations')
            extra_kwargs['tokens_per_sec'] = results.get('tokens_per_sec')
            extra_kwargs['train_loss'] = results.get('train_loss')
        
        # The upload_benchmark_results function will print the success message and ID.
        upload_benchmark_results(
            model_name=args.model,
            primary_metric_value=primary_metric_val, # This is now consistently from results.get('result')
            max_temp=max_temp_val,
            avg_temp=avg_temp_val,
            cloud_provider=provider,
            **extra_kwargs
        )
        
        print("Benchmark completed") # Final confirmation message
    elif results and results.get("error"):
        print(f"\nBenchmark failed: {results.get('error')}")
    elif results is None and args.model not in ["stable-diffusion-1-5", "qwen3-0-6b", "nanogpt-train"]:
        pass # Error already printed
    else:
        print("\nBenchmark was canceled or did not complete. Results not submitted.")
        if results:
            iterations_before_cancel = results.get('iterations', 0)
            if args.model == "qwen3-0-6b":
                print(f"Generations processed before cancellation: {results.get('result', 0)}")
            elif args.model == "stable-diffusion-1-5":
                print(f"Images generated before cancellation: {results.get('result', 0)}")
            elif args.model == "nanogpt-train":
                print(f"Training iterations before cancellation: {iterations_before_cancel}")

if __name__ == "__main__":
    main()