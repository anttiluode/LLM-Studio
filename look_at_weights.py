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


class LLMWeightVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Weight Visualizer")
        self.root.geometry("1200x800")
        
        # State
        self.weight_dict = {}
        self.weights_2d = None
        self.selection_rect = None
        
        # State for managing plots and color scales
        self.vmin = None
        self.vmax = None
        
        # Setup GUI
        self._setup_gui()
        
    def _setup_gui(self):
        # ... (GUI setup code remains the same) ...
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
        
        tk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=20)
        self.cmap_var = tk.StringVar(value='viridis')
        cmap_combo = ttk.Combobox(control_frame, textvariable=self.cmap_var,
                                 values=['viridis', 'plasma', 'coolwarm', 'RdBu', 
                                        'seismic', 'gray', 'hot', 'jet'],
                                 width=10, state='readonly')
        cmap_combo.pack(side=tk.LEFT, padx=5)
        cmap_combo.bind('<<ComboboxSelected>>', self.update_display)
        
        stats_frame = tk.Frame(self.root)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.stats_label = tk.Label(stats_frame, text="No weights loaded", 
                                   justify=tk.LEFT, font=('Courier', 10))
        self.stats_label.pack(side=tk.LEFT)
        
        display_frame = tk.Frame(self.root)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig1, self.ax1 = plt.subplots(figsize=(6, 6))
        self.canvas1 = FigureCanvasTkAgg(self.fig1, display_frame)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.fig2, self.ax2 = plt.subplots(figsize=(6, 6))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, display_frame)
        self.canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.rs = RectangleSelector(self.ax1, self.on_select,
                                   useblit=True, button=[1], minspanx=5, minspany=5,
                                   spancoords='pixels', interactive=True)
        
    def load_model(self):
        filename = filedialog.askopenfilename(
            title="Load PyTorch Model",
            filetypes=[("PyTorch files", "*.pt *.pth"), ("All files", "*.*")]
        )
        if not filename: return
        try:
            checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.weight_dict = {
                name: param.detach().cpu().numpy()
                for name, param in state_dict.items()
                if isinstance(param, torch.Tensor)
            }
            layer_names = [f"{name} {list(param.shape)}" for name, param in self.weight_dict.items()]
            self.layer_combo['values'] = layer_names
            if layer_names:
                self.layer_combo.current(0)
                self.on_layer_select(None)
            self.stats_label.config(text=f"Loaded {len(layer_names)} weight tensors")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def on_layer_select(self, event):
        if not self.layer_combo.get(): return
        layer_full = self.layer_combo.get()
        layer_name = layer_full.split(' [')[0]
        if layer_name in self.weight_dict:
            self.update_display(self.weight_dict[layer_name])

    def update_display(self, weights=None, event=None):
        if weights is None:
            if not self.layer_combo.get(): return
            layer_name = self.layer_combo.get().split(' [')[0]
            weights = self.weight_dict[layer_name]

        self.rs.set_active(False)
        self.fig1.clear()
        self.fig2.clear()
        self.ax1 = self.fig1.add_subplot(111)
        self.ax2 = self.fig2.add_subplot(111)

        if weights.ndim == 1:
            size = int(np.sqrt(len(weights))) + 1
            padded = np.zeros(size * size); padded[:len(weights)] = weights
            self.weights_2d = padded.reshape(size, size)
        elif weights.ndim > 2:
            self.weights_2d = weights.reshape(weights.shape[0], -1)
        else:
            self.weights_2d = weights

        cmap = self.cmap_var.get()
        im1 = self.ax1.imshow(self.weights_2d, cmap=cmap, interpolation='nearest', aspect='auto')
        self.vmin, self.vmax = im1.get_clim()
        self.ax1.set_title(f"Full Weight Matrix {self.weights_2d.shape}")
        self.fig1.colorbar(im1, ax=self.ax1)
        self.rs.ax = self.ax1 # Re-attach selector to the new axes

        im2 = self.ax2.imshow(self.weights_2d, cmap=cmap, interpolation='nearest', aspect='auto')
        self.ax2.set_title("Zoomed View (click and drag on left to zoom)")
        self.fig2.colorbar(im2, ax=self.ax2)
        
        stats_text = (f"Shape: {weights.shape}\n"
                      f"Min: {weights.min():.6f}, Max: {weights.max():.6f}\n"
                      f"Mean: {weights.mean():.6f}, Std: {weights.std():.6f}\n"
                      f"Zeros: {(weights == 0).sum()} ({(weights == 0).sum() / weights.size * 100:.1f}%)")
        self.stats_label.config(text=stats_text)

        self.canvas1.draw()
        self.canvas2.draw()
        self.rs.set_active(True)

    def on_select(self, eclick, erelease):
        if self.weights_2d is None: return
        
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        h, w = self.weights_2d.shape
        x1, x2 = sorted([max(0, min(x1, w-1)), max(0, min(x2, w-1))])
        y1, y2 = sorted([max(0, min(y1, h-1)), max(0, min(y2, h-1))])
        
        if x2 - x1 < 1 or y2 - y1 < 1: return

        zoomed = self.weights_2d[y1:y2+1, x1:x2+1]
        
        # --- START OF FIX ---
        # Completely clear the figure and redraw everything to prevent artifacts
        self.fig2.clear()
        self.ax2 = self.fig2.add_subplot(111)
        # --- END OF FIX ---

        im = self.ax2.imshow(zoomed, cmap=self.cmap_var.get(), 
                            interpolation='nearest', aspect='auto',
                            vmin=self.vmin, vmax=self.vmax)
        self.fig2.colorbar(im, ax=self.ax2) # Recreate the colorbar
        
        self.ax2.set_title(f"Zoomed: [{y1}:{y2+1}, {x1}:{x2+1}]")
        
        if zoomed.shape[0] <= 50 and zoomed.shape[1] <= 50:
            self.ax2.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
            self.ax2.set_xticks(np.arange(-.5, zoomed.shape[1], 1)); self.ax2.set_xticklabels([])
            self.ax2.set_yticks(np.arange(-.5, zoomed.shape[0], 1)); self.ax2.set_yticklabels([])

            if zoomed.shape[0] <= 15 and zoomed.shape[1] <= 15:
                for i in range(zoomed.shape[0]):
                    for j in range(zoomed.shape[1]):
                        val = zoomed[i, j]
                        color = 'white' if val < (self.vmin + self.vmax) / 2 else 'black'
                        self.ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)

        self.canvas2.draw()
        
        if self.selection_rect and self.selection_rect.axes: self.selection_rect.remove()
        self.selection_rect = patches.Rectangle((x1-0.5, y1-0.5), 
                                              x2-x1+1, y2-y1+1,
                                              linewidth=2, edgecolor='red', facecolor='none')
        self.ax1.add_patch(self.selection_rect)
        self.canvas1.draw()

def main():
    root = tk.Tk()
    app = LLMWeightVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
