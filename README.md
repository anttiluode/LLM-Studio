# Simple LLM Studio & Weight Visualizer

This project provides a simple, self-contained environment to train a GPT-style Large Language Model (LLM) from scratch and visually 
inspect its internal weights. It's designed for educational purposes to demonstrate the core components of LLM training and analysis 
in a hands-on way.

The project consists of two main components:

LLM Trainer Studio (llm_studio.py): A graphical user interface (GUI) for configuring, training, and testing a custom GPT model
on your own text data.

Weight Visualizer (look_at_weights.py): A tool to load a trained model and visualize its weight matrices as interactive,
zoomable heatmaps.

# Features

🧠 LLM Trainer Studio

Graphical Interface: Easy-to-use Tkinter GUI to manage the entire training process.

Custom GPT Model: Trains a standard decoder-only Transformer model built from scratch using PyTorch.

Configurable Architecture: Choose from "small," "medium," or "large" model sizes, or tweak the configuration yourself.

Flexible Training: Adjust hyperparameters like batch size, epochs, and learning rate.

Train on Your Data: Load any .txt file to serve as your training corpus.

Save & Load: Save your trained model checkpoints (.pt files) and load them back later to continue training or for inference.

Real-time Logging & Inference: Monitor the training progress in real-time and test your model with prompts directly in the UI.

# 🔬 Weight Visualizer

Interactive GUI: An intuitive interface built with Tkinter and Matplotlib.

Load Custom Models: Opens the .pt model files saved by the Trainer Studio.

Layer-by-Layer Inspection: A dropdown menu lets you select and view any weight or bias tensor in the model.

Zoomable Heatmaps: Click and drag on the main weight matrix to see a high-resolution view of any section.

Detailed Statistics: View key stats for each tensor, including its shape, min/max values, mean, and standard deviation.

# Screenshots

Here's a look at the two applications in action.

LLM Trainer Studio in action:

Inspecting model weights with the Visualizer:

🚀 Getting Started

Requirements

To run this project, you'll need Python 3 and the following libraries. It's highly recommended to use a virtual environment.

torch
transformers
numpy
matplotlib
You can install them all with pip:

pip install torch transformers numpy matplotlib

# Usage

Follow these steps to train your own model and visualize its weights.

Step 1: Prepare Your Training Data

Find a plain text (.txt) file to use for training. The larger, the better.

A good source is Project Gutenberg, where you can download books as plain text files.

Save this file somewhere you can easily find it.

Step 2: Train a Model with llm_studio.py

Run the trainer studio from your terminal:

python llm_studio.py

Load Data: Click the Load Training Data button and select the .txt file you prepared.

Initialize Model: Choose a model size and click Initialize Model. This will build the model in memory and prepare the tokenizer.

Configure Training: Adjust the batch size, epochs, and learning rate as needed. Smaller models can tolerate larger batch sizes.

Start Training: Click the Start Training button. The training will run in a separate thread so the UI remains responsive.
You can monitor the progress in the log.

Save Model: Once training is complete, click the Save Model button and save your checkpoint file (e.g., my_first_model.pt).

Step 3: Visualize the Weights with look_at_weights.py

Run the weight visualizer from your terminal:

python look_at_weights.py

Load Model: Click the Load Model (.pt) button and select the .pt file you just saved.

# Explore:

Use the Select Layer dropdown to choose a weight tensor to inspect.

Click and drag your mouse over a region on the "Full Weight Matrix" (left plot).

The "Zoomed View" (right plot) will update to show a detailed heatmap of your selection.

# How It Works

LLM Trainer Studio (llm_studio.py)
Model: The application implements a standard GPT (Generative Pre-trained Transformer) architecture in PyTorch. 
It includes multi-head self-attention with causal masking, feed-forward blocks, layer normalization, and weight
tying between the token embedding and the final linear layer.

Data Handling: It uses the transformers library to load a standard GPT-2 tokenizer. The input text file is
split into segments, tokenized, and prepared for causal language modeling, where the model learns to predict the next token in a sequence.

GUI & Threading: The UI is built with Python's standard tkinter library. The training process is launched in a separate
thread to prevent the GUI from freezing, allowing for real-time log updates.

Weight Visualizer (look_at_weights.py)
Model Loading: The script uses torch.load with weights_only=False. This is necessary because the saved checkpoint 
includes not just the model's numerical weights (state_dict) but also a pickled Python object: the GPTConfig class 
instance that defines the model's architecture. The script includes a copy of the GPTConfig class so it can
correctly unpickle and understand the model's structure.

Visualization: The UI embeds matplotlib plots into the tkinter window. It uses imshow to render the 2D
weight tensors as heatmaps. The interactive zoom is powered by Matplotlib's RectangleSelector widget,
which captures the coordinates of the user's selection and uses them to update the data shown in the second plot.

# Licence MIT
