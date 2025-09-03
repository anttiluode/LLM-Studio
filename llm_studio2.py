# memory_efficient_llm_trainer.py
# Complete LLM trainer with memory management for large datasets
# Requirements: torch, transformers, tkinter

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from pathlib import Path
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# -----------------------------
# Standard GPT Model (with memory optimizations)
# -----------------------------

class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,  # GPT-2 vocab size
        max_seq_len=1024,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
        dropout=0.1,
        bias=True,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.bias = bias

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len) * float('-inf'), diagonal=1)
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        att = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)
        att = att + self.causal_mask[:T, :T]
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return self.dropout(y)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_emb.weight = self.lm_head.weight
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.95):
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_crop = input_ids if input_ids.shape[1] <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
            
            logits, _ = self(input_ids_crop)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Optional top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        
        return input_ids

# -----------------------------
# Memory-Efficient Dataset
# -----------------------------

class MemoryEfficientTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512, max_samples=None, chunk_overlap=50):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_overlap = chunk_overlap
        
        # Limit dataset size if specified
        if max_samples and len(texts) > max_samples:
            print(f"Limiting dataset from {len(texts)} to {max_samples} samples for memory management")
            texts = texts[:max_samples]
        
        # Process texts into chunks more efficiently
        self.input_ids = []
        self._process_texts_chunked(texts)
        
        print(f"Created dataset with {len(self.input_ids)} samples")
    
    def _process_texts_chunked(self, texts):
        """Process texts in chunks to manage memory"""
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"Processing text {i}/{len(texts)}")
                
            # Tokenize text
            try:
                tokens = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids'][0]
            except:
                continue  # Skip problematic texts
            
            # Skip very short texts
            if len(tokens) < 10:
                continue
                
            # If text is longer than max_len, create overlapping chunks
            if len(tokens) > self.max_len:
                start = 0
                while start < len(tokens):
                    end = start + self.max_len
                    chunk = tokens[start:end]
                    if len(chunk) > 10:  # Only keep meaningful chunks
                        self.input_ids.append(chunk)
                    start += self.max_len - self.chunk_overlap
                    if end >= len(tokens):
                        break
            else:
                self.input_ids.append(tokens)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        # For language modeling, labels are shifted input_ids
        if len(tokens) > 1:
            return {
                'input_ids': tokens[:-1],
                'labels': tokens[1:]
            }
        else:
            # Fallback for single token
            return {
                'input_ids': tokens,
                'labels': tokens
            }

def memory_efficient_collate_fn(batch):
    """Collate function that handles variable lengths efficiently"""
    # Filter out None batches and empty items
    batch = [b for b in batch if b is not None and len(b['input_ids']) > 0]
    if not batch:
        return None
        
    # Find max length in this batch (not global max)
    max_len = max(len(item['input_ids']) for item in batch)
    # Cap at reasonable size to save memory
    max_len = min(max_len, 512)
    
    input_ids = []
    labels = []
    
    for item in batch:
        # Truncate if too long
        input_seq = item['input_ids'][:max_len]
        label_seq = item['labels'][:max_len]
        
        # Pad to batch max length
        pad_len = max_len - len(input_seq)
        if pad_len > 0:
            input_ids.append(F.pad(input_seq, (0, pad_len), value=0))
            labels.append(F.pad(label_seq, (0, pad_len), value=-100))
        else:
            input_ids.append(input_seq)
            labels.append(label_seq)
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }

# -----------------------------
# Memory-Efficient Training
# -----------------------------

def train_epoch_with_memory_management(model, dataloader, optimizer, scheduler, device, 
                                     gradient_accumulation_steps=1, log_fn=None):
    model.train()
    total_loss = 0
    valid_batches = 0
    accumulation_loss = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        # Skip None batches
        if batch is None:
            continue
            
        try:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass
            logits, loss = model(input_ids, labels)
            
            # Scale loss by accumulation steps
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            accumulation_loss += loss.item()
            
            # Step optimizer every gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                
                total_loss += accumulation_loss
                valid_batches += 1
                accumulation_loss = 0
            
            # Clear cache periodically
            if i % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if i % 50 == 0 and log_fn:
                current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
                log_fn(f"Step {i}/{len(dataloader)}: loss={loss.item()*gradient_accumulation_steps:.4f}, lr={current_lr:.2e}\n")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                if log_fn:
                    log_fn(f"OOM at step {i}, skipping batch...\n")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()  # Clear gradients after OOM
                continue
            else:
                raise e
    
    # Handle remaining gradients if any
    if accumulation_loss > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += accumulation_loss
        valid_batches += 1
    
    return total_loss / max(valid_batches, 1)

# -----------------------------
# Enhanced GUI with Memory Management
# -----------------------------

class MemoryEfficientLLMTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Memory-Efficient LLM Trainer")
        self.geometry("1100x900")
        
        # State
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training params with memory-friendly defaults
        self.model_size = tk.StringVar(value="small")
        self.batch_size = tk.IntVar(value=2)  # Reduced default
        self.epochs = tk.IntVar(value=1)
        self.lr = tk.DoubleVar(value=3e-5)
        self.max_len = tk.IntVar(value=256)  # Reduced default
        self.max_samples = tk.IntVar(value=10000)  # New: limit dataset size
        self.gradient_accumulation = tk.IntVar(value=4)  # New: gradient accumulation
        
        self._build_ui()
    
    def _build_ui(self):
        # Top frame
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(top, text="Model Size:").pack(side=tk.LEFT)
        ttk.Combobox(
            top, 
            textvariable=self.model_size,
            values=["tiny", "small", "medium"],  # Removed "large"
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top, text="Load Training Data", command=self._load_data, bg="#d3e3f3").pack(side=tk.LEFT, padx=20)
        tk.Button(top, text="Initialize Model", command=self._init_model, bg="#d3f3e3").pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Check GPU Memory", command=self._check_gpu_memory, bg="#f3e3d3").pack(side=tk.LEFT, padx=5)
        
        # Memory Management frame
        memory_frame = tk.LabelFrame(self, text="Memory Management")
        memory_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = tk.Frame(memory_frame)
        row.pack(fill=tk.X, pady=5)
        
        tk.Label(row, text="Max Samples:").pack(side=tk.LEFT, padx=5)
        tk.Entry(row, textvariable=self.max_samples, width=10).pack(side=tk.LEFT)
        
        tk.Label(row, text="Grad Accumulation:").pack(side=tk.LEFT, padx=20)
        tk.Entry(row, textvariable=self.gradient_accumulation, width=10).pack(side=tk.LEFT)
        
        # Add preset buttons for quick memory setup
        preset_frame = tk.Frame(memory_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="Low Memory", command=self._preset_low_memory, bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="Balanced", command=self._preset_balanced, bg="#ffffcc").pack(side=tk.LEFT, padx=5)
        tk.Button(preset_frame, text="High Memory", command=self._preset_high_memory, bg="#ccffcc").pack(side=tk.LEFT, padx=5)
        
        # Config frame
        config = tk.LabelFrame(self, text="Training Configuration")
        config.pack(fill=tk.X, padx=10, pady=5)
        
        row1 = tk.Frame(config)
        row1.pack(fill=tk.X, pady=5)
        
        tk.Label(row1, text="Batch Size:").pack(side=tk.LEFT, padx=5)
        tk.Entry(row1, textvariable=self.batch_size, width=10).pack(side=tk.LEFT)
        
        tk.Label(row1, text="Epochs:").pack(side=tk.LEFT, padx=20)
        tk.Entry(row1, textvariable=self.epochs, width=10).pack(side=tk.LEFT)
        
        tk.Label(row1, text="Learning Rate:").pack(side=tk.LEFT, padx=20)
        tk.Entry(row1, textvariable=self.lr, width=15).pack(side=tk.LEFT)
        
        tk.Label(row1, text="Max Length:").pack(side=tk.LEFT, padx=20)
        tk.Entry(row1, textvariable=self.max_len, width=10).pack(side=tk.LEFT)
        
        # Training controls
        train_frame = tk.Frame(self)
        train_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.train_btn = tk.Button(train_frame, text="Start Training", command=self._start_training, 
                                   bg="#90EE90", font=("Arial", 12, "bold"), state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT)
        
        self.save_btn = tk.Button(train_frame, text="Save Model", command=self._save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=20)
        
        tk.Button(train_frame, text="Load Model", command=self._load_model).pack(side=tk.LEFT)
        
        # GPU Memory display
        self.memory_label = tk.Label(self, text="GPU Memory: Unknown", font=('Courier', 10), bg="#f0f0f0")
        self.memory_label.pack(pady=5, fill=tk.X, padx=10)
        
        # Progress
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = tk.LabelFrame(self, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log = scrolledtext.ScrolledText(log_frame, height=15, font=('Courier', 9))
        self.log.pack(fill=tk.BOTH, expand=True)
        
        # Generation frame
        gen_frame = tk.LabelFrame(self, text="Test Generation")
        gen_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = tk.Frame(gen_frame)
        row.pack(fill=tk.X, pady=5)
        
        tk.Label(row, text="Prompt:").pack(side=tk.LEFT, padx=5)
        self.prompt_entry = tk.Entry(row, width=50)
        self.prompt_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.prompt_entry.insert(0, "Hello world")
        
        tk.Button(row, text="Generate", command=self._generate).pack(side=tk.LEFT, padx=5)
        
        self.gen_output = tk.Text(gen_frame, height=4, wrap=tk.WORD, font=('Arial', 10))
        self.gen_output.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial GPU check
        self._check_gpu_memory()
    
    def _preset_low_memory(self):
        """Settings for <6GB GPU"""
        self.model_size.set("tiny")
        self.batch_size.set(1)
        self.max_len.set(128)
        self.max_samples.set(5000)
        self.gradient_accumulation.set(8)
    
    def _preset_balanced(self):
        """Settings for 6-12GB GPU"""
        self.model_size.set("small")
        self.batch_size.set(2)
        self.max_len.set(256)
        self.max_samples.set(10000)
        self.gradient_accumulation.set(4)
    
    def _preset_high_memory(self):
        """Settings for >12GB GPU"""
        self.model_size.set("medium")
        self.batch_size.set(4)
        self.max_len.set(512)
        self.max_samples.set(20000)
        self.gradient_accumulation.set(2)
    
    def _check_gpu_memory(self):
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(0) / 1e9
            cached = torch.cuda.memory_reserved(0) / 1e9
            free = total_mem - cached
            
            color = "#ccffcc" if free > 4 else "#ffffcc" if free > 2 else "#ffcccc"
            self.memory_label.config(
                text=f"GPU: {allocated:.1f}GB used, {free:.1f}GB free, {total_mem:.1f}GB total",
                bg=color
            )
        else:
            self.memory_label.config(text="GPU Memory: CUDA not available", bg="#ffcccc")
    
    def _log(self, msg):
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)
        self.update()
    
    def _get_model_config(self):
        configs = {
            "tiny": GPTConfig(n_layers=4, d_model=256, n_heads=4, d_ff=1024, max_seq_len=256),
            "small": GPTConfig(n_layers=6, d_model=384, n_heads=6, d_ff=1536, max_seq_len=512),
            "medium": GPTConfig(n_layers=8, d_model=512, n_heads=8, d_ff=2048, max_seq_len=512)
        }
        return configs[self.model_size.get()]
    
    def _init_model(self):
        try:
            self._log("Initializing model and tokenizer...\n")
            
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Use GPT-2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            config = self._get_model_config()
            config.vocab_size = self.tokenizer.vocab_size
            config.max_seq_len = self.max_len.get()
            
            self.model = GPT(config).to(self.device)
            
            total_params = sum(p.numel() for p in self.model.parameters())
            self._log(f"Model initialized: {total_params/1e6:.1f}M parameters\n")
            self._log(f"Device: {self.device}\n")
            
            self.train_btn.config(state=tk.NORMAL)
            self._check_gpu_memory()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Error initializing model: {str(e)}\n")
    
    def _load_data(self):
        filepath = filedialog.askopenfilename(
            title="Select training text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self._log(f"Loading data from {os.path.basename(filepath)}...\n")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # More efficient text splitting
            texts = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(texts) < 100:  # If too few paragraphs, split by sentences
                import re
                texts = re.split(r'(?<=[.!?])\s+', text)
                texts = [t.strip() for t in texts if len(t.strip()) > 20]
            
            self._log(f"Found {len(texts)} text segments\n")
            
            # Limit dataset size for memory management
            max_samples = self.max_samples.get()
            if len(texts) > max_samples:
                self._log(f"Limiting dataset from {len(texts)} to {max_samples} samples for memory management\n")
                texts = texts[:max_samples]
            
            self.texts = texts
            self._log(f"Loaded {len(texts)} text segments\n")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._log(f"Error loading data: {str(e)}\n")
    
    def _start_training(self):
        if self.model is None:
            messagebox.showwarning("No model", "Please initialize a model first")
            return
        
        if not hasattr(self, 'texts'):
            messagebox.showwarning("No data", "Please load training data first")
            return
        
        self.progress.start(10)
        self.train_btn.config(state=tk.DISABLED)
        
        # Run training in thread
        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()
    
    def _train_worker(self):
        try:
            # Clear GPU cache before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Create memory-efficient dataset
            self._log("Creating memory-efficient dataset...\n")
            dataset = MemoryEfficientTextDataset(
                self.texts, 
                self.tokenizer, 
                self.max_len.get(),
                max_samples=self.max_samples.get()
            )
            
            # Use smaller batch size and enable memory-efficient collation
            effective_batch_size = max(1, self.batch_size.get())
            dataloader = DataLoader(
                dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                collate_fn=memory_efficient_collate_fn,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False  # Can cause memory issues
            )
            
            # Setup optimizer with gradient accumulation
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.lr.get(), 
                weight_decay=0.01,
                eps=1e-8
            )
            
            grad_accum_steps = self.gradient_accumulation.get()
            total_steps = len(dataloader) * self.epochs.get() // grad_accum_steps
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(0.1 * total_steps)),
                num_training_steps=total_steps
            )
            
            self._log(f"Starting training for {self.epochs.get()} epochs...\n")
            self._log(f"Dataset size: {len(dataset)} samples\n")
            self._log(f"Total batches: {len(dataloader)}\n")
            self._log(f"Effective batch size: {effective_batch_size * grad_accum_steps}\n")
            self._log(f"Gradient accumulation steps: {grad_accum_steps}\n")
            
            # Training loop with memory management
            for epoch in range(self.epochs.get()):
                self._log(f"\n--- Epoch {epoch+1}/{self.epochs.get()} ---\n")
                
                avg_loss = train_epoch_with_memory_management(
                    self.model, dataloader, optimizer, scheduler,
                    self.device, grad_accum_steps, log_fn=self._log
                )
                
                self._log(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}\n")
                self._check_gpu_memory()
            
            self._log("\nTraining completed!\n")
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self._log(f"\nError during training: {str(e)}\n")
            if "out of memory" in str(e).lower():
                self._log("Memory suggestions:\n")
                self._log("- Try the 'Low Memory' preset\n")
                self._log("- Reduce batch_size to 1\n")
                self._log("- Reduce max_length to 128\n")
                self._log("- Reduce max_samples to 5000\n")
                self._log("- Increase gradient_accumulation_steps\n")
        finally:
            self.progress.stop()
            self.train_btn.config(state=tk.NORMAL)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _save_model(self):
        if self.model is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.model.config,
                    'tokenizer_name': 'gpt2',
                    'training_args': {
                        'model_size': self.model_size.get(),
                        'max_len': self.max_len.get(),
                        'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 50257
                    }
                }, filepath)
                self._log(f"Model saved to {filepath}\n")
            except Exception as e:
                self._log(f"Error saving model: {str(e)}\n")
                messagebox.showerror("Save Error", str(e))
    
    def _load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            
            # Load tokenizer
            tokenizer_name = checkpoint.get('tokenizer_name', 'gpt2')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            config = checkpoint['config']
            self.model = GPT(config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Update UI with loaded settings if available
            if 'training_args' in checkpoint:
                args = checkpoint['training_args']
                if 'model_size' in args:
                    self.model_size.set(args['model_size'])
                if 'max_len' in args:
                    self.max_len.set(args['max_len'])
            
            total_params = sum(p.numel() for p in self.model.parameters())
            self._log(f"Model loaded from {os.path.basename(filepath)}\n")
            self._log(f"Parameters: {total_params/1e6:.1f}M\n")
            
            self.train_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self._check_gpu_memory()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self._log(f"Error loading model: {str(e)}\n")
    
    @torch.no_grad()
    def _generate(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("No model", "Please load or train a model first")
            return
        
        prompt = self.prompt_entry.get()
        if not prompt.strip():
            return
        
        self.gen_output.delete(1.0, tk.END)
        self.gen_output.insert(1.0, "Generating...")
        self.update()
        
        try:
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
            
            # Generate
            self.model.eval()
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            
            # Decode
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            self.gen_output.delete(1.0, tk.END)
            self.gen_output.insert(1.0, output_text)
            
        except Exception as e:
            self.gen_output.delete(1.0, tk.END)
            self.gen_output.insert(1.0, f"Error: {str(e)}")
            self._log(f"Generation error: {str(e)}\n")

if __name__ == "__main__":
    # Set memory fraction to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    app = MemoryEfficientLLMTrainer()
    app.mainloop()