# simple_llm_trainer.py
# A simplified LLM trainer with standard tokenization
# Requirements: torch, transformers, tkinter, datasets

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# -----------------------------
# Standard GPT Model
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
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
# Dataset
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Tokenize all texts
        self.input_ids = []
        for text in texts:
            tokens = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt')['input_ids'][0]
            if len(tokens) > 1:  # Skip empty
                self.input_ids.append(tokens)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        tokens = self.input_ids[idx]
        # For language modeling, labels are shifted input_ids
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:]
        }

def collate_fn(batch):
    # Pad sequences
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    
    for item in batch:
        # Pad with 0 (usually pad token)
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(F.pad(item['input_ids'], (0, pad_len), value=0))
        labels.append(F.pad(item['labels'], (0, pad_len), value=-100))  # -100 is ignored by loss
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }

# -----------------------------
# Training
# -----------------------------

def train_epoch(model, dataloader, optimizer, scheduler, device, log_fn=None):
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        logits, loss = model(input_ids, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        if i % 10 == 0 and log_fn:
            log_fn(f"Step {i}/{len(dataloader)}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}\n")
    
    return total_loss / len(dataloader)

# -----------------------------
# GUI
# -----------------------------

class LLMTrainer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simple LLM Trainer")
        self.geometry("900x700")
        
        # State
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training params
        self.model_size = tk.StringVar(value="small")
        self.batch_size = tk.IntVar(value=4)
        self.epochs = tk.IntVar(value=3)
        self.lr = tk.DoubleVar(value=5e-5)
        self.max_len = tk.IntVar(value=512)
        
        self._build_ui()
    
    def _build_ui(self):
        # Top frame
        top = tk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(top, text="Model Size:").pack(side=tk.LEFT)
        ttk.Combobox(
            top, 
            textvariable=self.model_size,
            values=["small", "medium", "large"],
            width=10,
            state="readonly"
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(top, text="Load Training Data", command=self._load_data, bg="#d3e3f3").pack(side=tk.LEFT, padx=20)
        tk.Button(top, text="Initialize Model", command=self._init_model, bg="#d3f3e3").pack(side=tk.LEFT, padx=5)
        
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
                                   bg="#90EE90", font=("Arial", 12, "bold"))
        self.train_btn.pack(side=tk.LEFT)
        
        self.save_btn = tk.Button(train_frame, text="Save Model", command=self._save_model, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=20)
        
        tk.Button(train_frame, text="Load Model", command=self._load_model).pack(side=tk.LEFT)
        
        # Progress
        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Log
        log_frame = tk.LabelFrame(self, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log = scrolledtext.ScrolledText(log_frame, height=15)
        self.log.pack(fill=tk.BOTH, expand=True)
        
        # Generation frame
        gen_frame = tk.LabelFrame(self, text="Test Generation")
        gen_frame.pack(fill=tk.X, padx=10, pady=5)
        
        row = tk.Frame(gen_frame)
        row.pack(fill=tk.X, pady=5)
        
        tk.Label(row, text="Prompt:").pack(side=tk.LEFT, padx=5)
        self.prompt_entry = tk.Entry(row, width=50)
        self.prompt_entry.pack(side=tk.LEFT, padx=5)
        self.prompt_entry.insert(0, "Once upon a time")
        
        tk.Button(row, text="Generate", command=self._generate).pack(side=tk.LEFT, padx=5)
        
        self.gen_output = tk.Text(gen_frame, height=3, wrap=tk.WORD)
        self.gen_output.pack(fill=tk.X, padx=5, pady=5)
    
    def _log(self, msg):
        self.log.insert(tk.END, msg)
        self.log.see(tk.END)
        self.update()
    
    def _get_model_config(self):
        configs = {
            "small": GPTConfig(n_layers=6, d_model=384, n_heads=6, d_ff=1536),
            "medium": GPTConfig(n_layers=12, d_model=768, n_heads=12, d_ff=3072),
            "large": GPTConfig(n_layers=24, d_model=1024, n_heads=16, d_ff=4096)
        }
        return configs[self.model_size.get()]
    
    def _init_model(self):
        try:
            self._log("Initializing model and tokenizer...\n")
            
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
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _load_data(self):
        filepath = filedialog.askopenfilename(
            title="Select training text file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Simple splitting by paragraphs or sentences
            texts = [p.strip() for p in text.split('\n\n') if p.strip()]
            if len(texts) < 10:  # If too few paragraphs, split by sentences
                import re
                texts = re.split(r'(?<=[.!?])\s+', text)
                texts = [t.strip() for t in texts if len(t.strip()) > 10]
            
            self._log(f"Loaded {len(texts)} text segments from {os.path.basename(filepath)}\n")
            self.texts = texts
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
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
            # Create dataset
            self._log("Creating dataset...\n")
            dataset = TextDataset(self.texts, self.tokenizer, self.max_len.get())
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size.get(),
                shuffle=True,
                collate_fn=collate_fn
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr.get())
            total_steps = len(dataloader) * self.epochs.get()
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            
            self._log(f"Starting training for {self.epochs.get()} epochs...\n")
            self._log(f"Total batches: {len(dataloader)}\n")
            
            # Training loop
            for epoch in range(self.epochs.get()):
                self._log(f"\n--- Epoch {epoch+1}/{self.epochs.get()} ---\n")
                
                avg_loss = train_epoch(
                    self.model, dataloader, optimizer, scheduler,
                    self.device, log_fn=self._log
                )
                
                self._log(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}\n")
            
            self._log("\nTraining completed!\n")
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self._log(f"\nError during training: {str(e)}\n")
        finally:
            self.progress.stop()
            self.train_btn.config(state=tk.NORMAL)
    
    def _save_model(self):
        if self.model is None:
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if filepath:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model.config,
                'tokenizer_name': 'gpt2'
            }, filepath)
            self._log(f"Model saved to {filepath}\n")
    
    def _load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint.get('tokenizer_name', 'gpt2'))
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            config = checkpoint['config']
            self.model = GPT(config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self._log(f"Model loaded from {filepath}\n")
            self.train_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    @torch.no_grad()
    def _generate(self):
        if self.model is None or self.tokenizer is None:
            messagebox.showwarning("No model", "Please load or train a model first")
            return
        
        prompt = self.prompt_entry.get()
        if not prompt:
            return
        
        self.gen_output.delete(1.0, tk.END)
        self.gen_output.insert(1.0, "Generating...")
        self.update()
        
        try:
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
            
            # Generate
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=50,
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

if __name__ == "__main__":
    app = LLMTrainer()
    app.mainloop()