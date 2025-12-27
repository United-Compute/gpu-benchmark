# src/gpu_benchmark/benchmarks/nanogpt_train.py
"""
NanoGPT Training Benchmark

Trains a GPT-2 style model on Shakespeare for 5 minutes.
Primary metric: validation loss (lower = better GPU efficiency)

Model architecture from https://github.com/karpathy/nanoGPT
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os
import platform
from tqdm import tqdm
import pynvml
from dataclasses import dataclass
import tiktoken
import requests

# ============================================================================
# GPT Model (embedded from nanoGPT)
# ============================================================================

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304  # GPT-2 vocab size rounded up for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = device_type == 'cuda' and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer


# ============================================================================
# Data Preparation
# ============================================================================

def get_cache_dir():
    """Get the cache directory for storing prepared data."""
    cache_dir = os.path.expanduser("~/.cache/gpu_benchmark/shakespeare")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def prepare_data():
    """Download and tokenize Shakespeare dataset. Returns (train_data, val_data) as tensors."""
    cache_dir = get_cache_dir()
    train_path = os.path.join(cache_dir, "train.pt")
    val_path = os.path.join(cache_dir, "val.pt")
    
    # Check if already cached
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_data = torch.load(train_path, weights_only=True)
        val_data = torch.load(val_path, weights_only=True)
        return train_data, val_data
    
    # Download Shakespeare
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text_path = os.path.join(cache_dir, "input.txt")
    
    if not os.path.exists(text_path):
        print("Downloading Shakespeare dataset...")
        response = requests.get(data_url)
        response.raise_for_status()
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Tokenize using GPT-2 BPE
    print("Tokenizing dataset...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    
    # Split 90/10
    n = int(0.9 * len(tokens))
    train_data = tokens[:n]
    val_data = tokens[n:]
    
    # Cache
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    return train_data, val_data


# ============================================================================
# Training Utilities
# ============================================================================

def get_batch(data, block_size, batch_size, device):
    """Get a random batch of training data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=50):
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ============================================================================
# Platform Utilities (shared with other benchmarks)
# ============================================================================

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


def get_nvml_device_handle():
    """Get the correct NVML device handle for the GPU being used."""
    pynvml.nvmlInit()
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        try:
            original_gpu_index = int(cuda_visible_devices.split(',')[0])
            handle = pynvml.nvmlDeviceGetHandleByIndex(original_gpu_index)
            return handle
        except (ValueError, IndexError):
            print(f"Warning: Could not parse CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    
    cuda_idx = torch.cuda.current_device()
    return pynvml.nvmlDeviceGetHandleByIndex(cuda_idx)


# ============================================================================
# Main Benchmark Functions
# ============================================================================

# Training hyperparameters (fixed for reproducibility)
TRAIN_CONFIG = {
    'batch_size': 8,
    'block_size': 256,
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'warmup_iters': 100,
    'min_lr': 6e-5,
    'use_compile': True,  # Will fallback if not supported
}


def get_lr(it, warmup_iters, learning_rate, min_lr, max_iters):
    """Learning rate schedule with warmup and cosine decay."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def load_pipeline():
    """Load model, data, and return everything needed for training."""
    # Set seeds for reproducibility
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Use bfloat16 if available, otherwise float16
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    if device_type == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        dtype = torch.float16
        print("Using float16")
    
    # Prepare data
    train_data, val_data = prepare_data()
    
    # Create model
    config = GPTConfig(
        block_size=TRAIN_CONFIG['block_size'],
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config)
    model = model.to(device)
    
    # Try to compile for speed (PyTorch 2.0+)
    compiled = False
    if TRAIN_CONFIG['use_compile'] and hasattr(torch, 'compile'):
        try:
            print("Compiling model (this may take a minute on first run)...")
            model = torch.compile(model)
            compiled = True
        except Exception as e:
            print(f"torch.compile not available, using eager mode: {e}")
    
    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=TRAIN_CONFIG['weight_decay'],
        learning_rate=TRAIN_CONFIG['learning_rate'],
        betas=(TRAIN_CONFIG['beta1'], TRAIN_CONFIG['beta2']),
        device_type='cuda'
    )
    
    # Create gradient scaler for mixed precision
    # Only enabled for float16, as bfloat16 doesn't need scaling
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == torch.float16))
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scaler': scaler,
        'train_data': train_data,
        'val_data': val_data,
        'device': device,
        'dtype': dtype,
        'compiled': compiled,
        'config': config
    }


def run_benchmark(pipeline, duration):
    """Run the training benchmark for the specified duration in seconds."""
    model = pipeline['model']
    optimizer = pipeline['optimizer']
    scaler = pipeline['scaler']
    train_data = pipeline['train_data']
    val_data = pipeline['val_data']
    device = pipeline['device']
    dtype = pipeline['dtype']
    config = pipeline['config']
    
    batch_size = TRAIN_CONFIG['batch_size']
    block_size = TRAIN_CONFIG['block_size']
    grad_clip = TRAIN_CONFIG['grad_clip']
    learning_rate = TRAIN_CONFIG['learning_rate']
    min_lr = TRAIN_CONFIG['min_lr']
    warmup_iters = TRAIN_CONFIG['warmup_iters']
    
    # Estimate max iterations (rough, for LR schedule)
    # Assume ~100ms per iter, 300s duration = ~3000 iters
    max_iters = 5000
    
    handle = get_nvml_device_handle()
    
    # Tracking variables
    iteration = 0
    total_tokens = 0
    temp_readings = []
    power_readings = []
    train_losses = []
    val_losses = []
    
    # Warmup (not counted)
    print("Warming up (5 iterations)...")
    model.train()
    for _ in range(5):
        X, Y = get_batch(train_data, block_size, batch_size, device)
        with torch.amp.autocast('cuda', dtype=dtype):
            _, loss = model(X, Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Initial loss estimation
    print("Estimating initial loss...")
    losses = estimate_loss(model, train_data, val_data, block_size, batch_size, device)
    initial_val_loss = losses['val']
    print(f"Initial val loss: {initial_val_loss:.4f}")
    
    # Main training loop
    start_time = time.time()
    end_time = start_time + duration
    last_eval_iter = 0
    eval_interval = 100
    
    try:
        with tqdm(total=100, desc="Training progress", unit="%", ncols=100) as pbar:
            last_update_percent = 0
            
            while time.time() < end_time:
                # Sample GPU metrics
                try:
                    current_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_readings.append(current_temp)
                except:
                    current_temp = 0
                
                try:
                    current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    power_readings.append(current_power)
                except:
                    pass
                
                # Update learning rate
                lr = get_lr(iteration, warmup_iters, learning_rate, min_lr, max_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Get batch and train
                X, Y = get_batch(train_data, block_size, batch_size, device)
                
                with torch.amp.autocast('cuda', dtype=dtype):
                    _, loss = model(X, Y)
                
                train_losses.append(loss.item())
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                iteration += 1
                total_tokens += batch_size * block_size
                
                # Periodic evaluation
                if iteration - last_eval_iter >= eval_interval:
                    losses = estimate_loss(model, train_data, val_data, block_size, batch_size, device)
                    val_losses.append(losses['val'])
                    last_eval_iter = iteration
                
                # Update progress bar
                current_time = time.time()
                current_percent = min(100, int((current_time - start_time) / duration * 100))
                if current_percent > last_update_percent:
                    pbar.update(current_percent - last_update_percent)
                    pbar.set_postfix({
                        'iter': iteration,
                        'loss': f"{loss.item():.3f}",
                        'temp': f"{current_temp}°C"
                    }, refresh=True)
                    last_update_percent = current_percent
        
        # Final evaluation
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        print("\nCalculating final validation loss...")
        final_losses = estimate_loss(model, train_data, val_data, block_size, batch_size, device, eval_iters=100)
        final_val_loss = final_losses['val']
        final_train_loss = final_losses['train']
        
        # Calculate metrics
        tokens_per_sec = total_tokens / elapsed
        avg_temp = sum(temp_readings) / len(temp_readings) if temp_readings else 0
        max_temp = max(temp_readings) if temp_readings else 0
        avg_power = sum(power_readings) / len(power_readings) if power_readings else None
        
        # GPU memory
        try:
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = round(meminfo.total / (1024 * 1024 * 1024), 2)
        except:
            gpu_memory_total = None
        
        pynvml.nvmlShutdown()
        
        return {
            "completed": True,
            "result": round(final_val_loss, 4),  # Primary metric: validation loss
            "train_loss": round(final_train_loss, 4),
            "initial_val_loss": round(initial_val_loss, 4),
            "iterations": iteration,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "max_temp": max_temp,
            "avg_temp": round(avg_temp, 1),
            "elapsed_time": round(elapsed, 2),
            "gpu_power_watts": round(avg_power, 2) if avg_power else None,
            "gpu_memory_total": gpu_memory_total,
            "platform": get_clean_platform(),
            "acceleration": f"CUDA {torch.version.cuda}" if torch.cuda.is_available() else "N/A",
            "torch_version": torch.__version__,
            "compiled": pipeline['compiled']
        }
    
    except KeyboardInterrupt:
        pynvml.nvmlShutdown()
        return {
            "completed": False,
            "result": None,
            "iterations": iteration,
            "max_temp": max(temp_readings) if temp_readings else 0,
            "avg_temp": sum(temp_readings) / len(temp_readings) if temp_readings else 0
        }
    except Exception as e:
        pynvml.nvmlShutdown()
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return {
            "completed": False,
            "error": str(e),
            "result": None,
            "iterations": iteration
        }

