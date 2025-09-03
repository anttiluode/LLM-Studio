# llm_weight_visualizer.py
# Visualize LLM weights as zoomable images

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches

# --- START OF FIX ---
# Copied from llm_studio.py to allow loading the model's config object
class GPTConfig:
    def __init__(
        self,
        vocab_size=50257,
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
# --- END OF FIX ---


class LLMWeightVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Weight Visualizer")
        self.root.geometry("1200x800")
        
        # State
        self.model = None
        self.current_weights = None
        self.weight_dict = {}
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        tk.Button(control_frame, text="Load Model (.pt)", 
                 command=self.load_model, bg="#4CAF50").pack(side=tk.LEFT, padx=5)
        
        tk.Label(control_frame, text="Select Layer:").pack(side=tk.LEFT, padx=20)
        self.layer_var = tk.StringVar()
        self.layer_combo = ttk.Combobox(control_frame, textvariable=self.layer_var, 
                                       width=40, state='readonly')
        self.layer_combo.pack(side=tk.LEFT, padx=5)
        self.layer_combo.bind('<<ComboboxSelected>>', self.on_layer_select)
        
        # Colormap selection
        tk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=20)
        self.cmap_var = tk.StringVar(value='viridis')
        cmap_combo = ttk.Combobox(control_frame, textvariable=self.cmap_var,
                                 values=['viridis', 'plasma', 'coolwarm', 'RdBu', 
                                        'seismic', 'gray', 'hot', 'jet'],
                                 width=10, state='readonly')
        cmap_combo.pack(side=tk.LEFT, padx=5)
        cmap_combo.bind('<<ComboboxSelected>>', self.update_display)
        
        # Stats frame
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.stats_label = tk.Label(stats_frame, text="No weights loaded", 
                                   justify=tk.LEFT, font=('Courier', 10))
        self.stats_label.pack(side=tk.LEFT)
        
        # Main display area with two plots
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Full weight matrix
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 6))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, display_frame)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right: Zoomed view
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, display_frame)
        self.canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Rectangle selector for zooming
        self.rs = RectangleSelector(self.ax1, self.on_select,
                                   useblit=True,
                                   button=[1],  # left button
                                   minspanx=5, minspany=5,
                                   spancoords='pixels',
                                   interactive=True)
        
    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Load PyTorch Model",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            # --- START OF FIX ---
            # Load checkpoint using weights_only=False because the file contains
            # a pickled GPTConfig object, not just weights.
            checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            # --- END OF FIX ---
            
            # Extract state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint
            else:
                # Assume it's directly a state dict
                state_dict = checkpoint
            
            # Store weights
            self.weight_dict = {}
            layer_names = []
            
            for name, param in state_dict.items():
                if isinstance(param, torch.Tensor):
                    self.weight_dict[name] = param.detach().cpu().numpy()
                    layer_names.append(f"{name} {list(param.shape)}")
            
            # Update layer selector
            self.layer_combo['values'] = layer_names
            if layer_names:
                self.layer_combo.current(0)
                self.on_layer_select(None)
                
            self.stats_label.config(text=f"Loaded {len(layer_names)} weight tensors")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def on_layer_select(self, event):
        if not self.layer_combo.get():
            return
            
        # Extract layer name from combo selection
        layer_full = self.layer_combo.get()
        layer_name = layer_full.split(' [')[0]  # Remove shape info
        
        if layer_name in self.weight_dict:
            self.current_weights = self.weight_dict[layer_name]
            self.update_display()
    
    def update_display(self, event=None):
        if self.current_weights is None:
            return
        
        weights = self.current_weights
        cmap = self.cmap_var.get()
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Reshape weights for visualization
        if weights.ndim == 1:
            # 1D weights - reshape to square-ish
            size = int(np.sqrt(len(weights))) + 1
            padded = np.zeros(size * size)
            padded[:len(weights)] = weights
            weights_2d = padded.reshape(size, size)
        elif weights.ndim == 2:
            weights_2d = weights
        elif weights.ndim == 3:
            # For conv layers, show first filter
            weights_2d = weights[0]
        elif weights.ndim == 4:
            # For conv layers with multiple filters, flatten to 2D grid
            n_filters = weights.shape[0]
            grid_size = int(np.sqrt(n_filters)) + 1
            h, w = weights.shape[2], weights.shape[3]
            
            grid = np.zeros((grid_size * h, grid_size * w))
            for i in range(min(n_filters, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size
                if weights.shape[1] == 1:  # Single channel
                    grid[row*h:(row+1)*h, col*w:(col+1)*w] = weights[i, 0]
                else:  # Multiple channels - take mean
                    grid[row*h:(row+1)*h, col*w:(col+1)*w] = weights[i].mean(axis=0)
            weights_2d = grid
        else:
            # For higher dimensions, flatten to 2D
            weights_2d = weights.reshape(weights.shape[0], -1)
        
        # Display full weight matrix
        im1 = self.ax1.imshow(weights_2d, cmap=cmap, interpolation='nearest')
        self.ax1.set_title(f"Full Weight Matrix {weights_2d.shape}")
        self.ax1.set_xlabel("Weight Index")
        self.ax1.set_ylabel("Weight Index")
        self.fig1.colorbar(im1, ax=self.ax1)
        
        # Display initial zoomed view (full matrix)
        im2 = self.ax2.imshow(weights_2d, cmap=cmap, interpolation='nearest')
        self.ax2.set_title("Zoomed View (click and drag on left to zoom)")
        self.fig2.colorbar(im2, ax=self.ax2)
        
        # Update statistics
        stats_text = f"Shape: {weights.shape}\n"
        stats_text += f"Min: {weights.min():.6f}\n"
        stats_text += f"Max: {weights.max():.6f}\n"
        stats_text += f"Mean: {weights.mean():.6f}\n"
        stats_text += f"Std: {weights.std():.6f}\n"
        stats_text += f"Zeros: {(weights == 0).sum()} ({(weights == 0).sum() / weights.size * 100:.1f}%)"
        
        self.stats_label.config(text=stats_text)
        
        # Store for zooming
        self.weights_2d = weights_2d
        
        self.canvas1.draw()
        self.canvas2.draw()
    
    def on_select(self, eclick, erelease):
        if self.weights_2d is None:
            return
            
        # Get selection bounds
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure bounds are within image
        h, w = self.weights_2d.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        # Ensure proper ordering
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        # Extract zoomed region
        zoomed = self.weights_2d[y1:y2+1, x1:x2+1]
        
        # Update zoomed view
        self.ax2.clear()
        im = self.ax2.imshow(zoomed, cmap=self.cmap_var.get(), 
                            interpolation='nearest', aspect='auto')
        self.ax2.set_title(f"Zoomed: [{y1}:{y2+1}, {x1}:{x2+1}]")
        
        # Add grid for detailed view if zoomed enough
        if zoomed.shape[0] < 50 and zoomed.shape[1] < 50:
            self.ax2.set_xticks(np.arange(-0.5, zoomed.shape[1], 1), minor=True)
            self.ax2.set_yticks(np.arange(-0.5, zoomed.shape[0], 1), minor=True)
            self.ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            # Add value annotations if really zoomed
            if zoomed.shape[0] < 10 and zoomed.shape[1] < 10:
                for i in range(zoomed.shape[0]):
                    for j in range(zoomed.shape[1]):
                        val = zoomed[i, j]
                        color = 'white' if val < zoomed.mean() else 'black'
                        self.ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                                     color=color, fontsize=8)
        
        # Update colorbar
        self.fig2.colorbar(im, ax=self.ax2)
        self.canvas2.draw()
        
        # Highlight selection on main plot
        if hasattr(self, 'selection_rect'):
            self.selection_rect.remove()
        self.selection_rect = patches.Rectangle((x1-0.5, y1-0.5), 
                                              x2-x1+1, y2-y1+1,
                                              linewidth=2, edgecolor='red',
                                              facecolor='none')
        self.ax1.add_patch(self.selection_rect)
        self.canvas1.draw()

def main():
    root = tk.Tk()
    app = LLMWeightVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()