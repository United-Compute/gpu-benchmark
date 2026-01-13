# src/gpu_benchmark/benchmark.py
import torch
import time
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
import platform
import os

if hasattr(torch.version, "hip"):
    gpuType = "AMD"
    import amdsmi
else:
    gpuType = "Nvidia"
    import pynvml

def get_clean_platform():
    os_platform = platform.system()
    if os_platform == "Linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.strip().split("=")[1].strip('"')
        except Exception:
            pass
        return f"Linux {platform.release()}"
    elif os_platform == "Windows":
        return f"Windows {platform.release()}"
    elif os_platform == "Darwin":
        return f"macOS {platform.mac_ver()[0]}"
    else:
        return os_platform

def load_pipeline():
    """Load the Stable Diffusion pipeline and return it."""    
    model_id = "yachty66/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipe.to("cuda")
    return pipe

def get_nvml_device_handle():
    """Get the correct NVML device handle for the GPU being used."""
    if gpuType == "AMD":
        amdsmi.amdsmi_init()
    else:
        pynvml.nvmlInit()

    # Check CUDA_VISIBLE_DEVICES first
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        try:
            # When CUDA_VISIBLE_DEVICES is set, the first (and only) visible GPU
            # becomes index 0 to CUDA, but we need to use the original index for NVML
            original_gpu_index = int(cuda_visible_devices.split(',')[0])
            if gpuType == "AMD":
                handle = amdsmi.amdsmi_get_processor_handles()[original_gpu_index]
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(original_gpu_index)
            return handle
        except (ValueError, IndexError):
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    
    # Fallback to current CUDA device
    cuda_idx = torch.cuda.current_device()
    if gpuType == "AMD":
        return amdsmi.amdsmi_get_processor_handles()[cuda_idx]
    else:
        return pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)

def getTemperature(handle, isFinalTemp):
    if gpuType == "AMD":
        try:
            return amdsmi.amdsmi_get_temp_metric(handle, amdsmi.AmdSmiTemperatureType.EDGE, amdsmi.AmdSmiTemperatureMetric.CURRENT)
        except amdsmi.AmdSmiException as e:
            if "Uninitialized" not in str(e) or torch.cuda.is_available():
                if isFinalTemp:
                    print(f"AMDSMI warning (temperature): {e}")
                else:
                    print(f"NVML warning (final temperature): {e}")
    else:
        try:
            return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except pynvml.NVMLError as e:
            if "Uninitialized" not in str(e) or torch.cuda.is_available(): # Log if not expected
                if isFinalTemp:
                    print(f"AMDSMI warning (temperature): {e}")
                else:
                    print(f"NVML warning (final temperature): {e}")

def getPowerUsage(handle):
    if gpuType == "AMD":
        try:
            return amdsmi.amdsmi_get_power_info(handle)['average_socket_power']
        except amdsmi.AmdSmiException as e:
            if "Uninitialized" not in str(e) or torch.cuda.is_available():
                print(f"AMDSMI warning (power): {e}")
    else:
        try:
            return round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 2)  # mW to W 2 decimal places
        except pynvml.NVMLError as e:
            if "Uninitialized" not in str(e) and "Not Supported" not in str(e) or torch.cuda.is_available():
                    print(f"NVML warning (power): {e}")

def getMemoryInfo(handle):
    if gpuType == "AMD":
        try:
            return round(amdsmi.amdsmi_get_gpu_memory_total(handle, amdsmi.amdsmi_interface.AmdSmiMemoryType.VRAM) / (1024**3), 2) # bytes to GB
        except amdsmi.AmdSmiException as e:
            if "Uninitialized" not in str(e) or torch.cuda.is_available():
                print(f"AMDSMI warning (memory info): {e}")
    else:
        try:
            return round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2) # bytes to GB
        except pynvml.NVMLError as e:
            if "Uninitialized" not in str(e) or torch.cuda.is_available():
                print(f"NVML warning (memory info): {e}")

def handleCleanup():
    if gpuType == "AMD":
        amdsmi.amdsmi_shut_down()
    else:
        pynvml.nvmlShutdown()

def run_benchmark(pipe, duration):
    """Run the GPU benchmark for the specified duration in seconds."""

    # Get the correct NVML handle for the GPU being used
    handle = get_nvml_device_handle()

    # Setup variables
    image_count = 0
    total_gpu_time = 0
    temp_readings = []
    power_readings = []

    # Start benchmark
    start_time = time.time()
    end_time = start_time + duration
    prompt = "a photo of an astronaut riding a horse on mars"

    try:
        # Disable progress bar for the pipeline
        pipe.set_progress_bar_config(disable=True)

        # Create a progress bar for the entire benchmark
        with tqdm(total=100, desc="Benchmark progress", unit="%") as pbar:
            # Calculate update amount per check
            last_update_time = start_time
            last_update_percent = 0

            # Run until time is up
            while time.time() < end_time:
                # Get GPU temperature
                current_temp = getTemperature(handle, False)
                temp_readings.append(current_temp)

                # CUDA timing events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()

                # Record start time and generate image
                start_event.record()
                image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                end_event.record()
                torch.cuda.synchronize()

                # Calculate timing
                gpu_time_ms = start_event.elapsed_time(end_event)
                total_gpu_time += gpu_time_ms

                # Update counter
                image_count += 1

                # Sample power usage
                try:
                    current_power = getPowerUsage(handle)
                    power_readings.append(current_power)
                except:
                    pass

                # Update progress bar
                current_time = time.time()
                current_percent = min(100, int((current_time - start_time) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    pbar.set_postfix({
                        'Images': image_count,
                        'Temp': f"{current_temp}°C"
                    })
                    last_update_percent = current_percent

        # Final temperature reading
        final_temp = getTemperature(handle, True)
        temp_readings.append(final_temp)

        # Calculate results
        elapsed = time.time() - start_time
        avg_time_ms = total_gpu_time / image_count if image_count > 0 else 0
        avg_temp = sum(temp_readings) / len(temp_readings)
        max_temp = max(temp_readings)

        # Get GPU power info
        try:
            power_usage = getPowerUsage(handle)
        except:
            power_usage = None

        # Get GPU memory info
        try:
            gpu_memory_total = getMemoryInfo(handle)
        except:
            gpu_memory_total = None

        # Get platform info
        platform_info = get_clean_platform()

        # Get CUDA version (acceleration)
        if gpuType == "AMD":
            cuda_version = f"ROCM {torch.version.hip}" if torch.cuda.is_available() else "N/A"
        else:
            cuda_version = f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A"

        # Get torch version
        torch_version = torch.__version__

        # Clean up
        handleCleanup()

        # Calculate average power
        avg_power = round(sum(power_readings) / len(power_readings), 2) if power_readings else None

        # Return benchmark results with completed flag
        return {
            "completed": True,  # Flag indicating the benchmark completed successfully
            "result": image_count,
            "max_temp": max_temp,
            "avg_temp": avg_temp,
            "elapsed_time": elapsed,
            "avg_time_ms": avg_time_ms,
            "gpu_utilization": (total_gpu_time/1000)/elapsed*100,
            "gpu_power_watts": avg_power,
            "gpu_memory_total": gpu_memory_total,
            "platform": platform_info,
            "acceleration": cuda_version,
            "torch_version": torch_version
        }

    except KeyboardInterrupt:
        # Clean up and return partial results with completed flag set to False
        handleCleanup()
        return {
            "completed": False,  # Flag indicating the benchmark was canceled
            "result": image_count,
            "max_temp": max(temp_readings) if temp_readings else 0,
            "avg_temp": sum(temp_readings)/len(temp_readings) if temp_readings else 0
        }
    except Exception as e:
        # Handle any other errors, clean up, and return error info
        handleCleanup()
        print(f"Error during benchmark: {e}")
        return {
            "completed": False,  # Flag indicating the benchmark failed
            "error": str(e),
            "result": image_count
        }
