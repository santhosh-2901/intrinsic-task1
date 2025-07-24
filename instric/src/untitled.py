import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add your 'src' folder to the system path
sys.path.append('/content/unzipped_folder/instric/src')

# Import your custom modules now that the path is known
from dataset import TextMelDataset, TextMelCollate
from model import Tacotron2
from torch.utils.data import DataLoader

# Setup Matplotlib for cleaner plots
plt.style.use('seaborn-v0_8-whitegrid')

# Set the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Dataset ---
# We need the dataset to know the vocabulary size for our model
dataset = TextMelDataset("toy_tts_parallel_data.csv")
n_vocab = len(dataset.vocab)
print(f"Vocabulary size: {n_vocab} characters")


# --- Load Trained Model ---
# Initialize the model architecture
model = Tacotron2(n_vocab=n_vocab)

# Load the saved state dictionary
try:
    model.load_state_dict(torch.load('tts_model.pth', map_location=device))
except FileNotFoundError:
    print("ERROR: tts_model.pth not found. Make sure you have trained the model first.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")


# Move the model to the selected device and set to evaluation mode
model.to(device)
model.eval()

print("Model loaded successfully and set to evaluation mode.")
# --- Prepare a single data sample for inference ---
collate_fn = TextMelCollate()
# We use a DataLoader to easily get a padded batch
demo_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
text_batch, mel_batch = next(iter(demo_loader))

# Select a single sample from the batch and move to the device
sample_text = text_batch[0:1].to(device)
ground_truth_mel = mel_batch[0:1].to(device)


# --- Run Inference ---
with torch.no_grad():
    # The model expects the target mel for teacher-forcing during training,
    # but we provide it here even in eval mode as the architecture requires it.
    predicted_mel = model(sample_text, ground_truth_mel)


# --- Visualize the Comparison ---
# Move tensors to CPU for plotting with numpy/matplotlib
ground_truth_mel_np = ground_truth_mel.squeeze().cpu().numpy()
predicted_mel_np = predicted_mel.squeeze().cpu().numpy()


# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot the Ground Truth
im1 = axes[0].imshow(ground_truth_mel_np.T, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title("Ground Truth Mel Spectrogram", fontsize=14)
axes[0].set_ylabel("Mel Bins", fontsize=12)
fig.colorbar(im1, ax=axes[0], format='%+2.0f dB')


# Plot the Predicted Spectrogram
im2 = axes[1].imshow(predicted_mel_np.T, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title("Predicted Mel Spectrogram (from Model)", fontsize=14)
axes[1].set_xlabel("Time Frames", fontsize=12)
axes[1].set_ylabel("Mel Bins", fontsize=12)
fig.colorbar(im2, ax=axes[1], format='%+2.0f dB')

plt.tight_layout(pad=2.0)
plt.show()