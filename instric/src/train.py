import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TextMelDataset, TextMelCollate
from model import Tacotron2

def train(model, dataloader, device, epochs=200, learning_rate=0.001): # <-- Added 'device'
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()

    print(f"Starting training on device: {device}") # <-- Added print statement
    for epoch in range(epochs):
        for i, (text, mel) in enumerate(dataloader):
            # --- Move data to the selected device (GPU or CPU) ---
            text = text.to(device) # <-- NEW
            mel = mel.to(device)   # <-- NEW

            optimizer.zero_grad()
            output = model(text, mel)
            loss = criterion(output, mel)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if (i + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print("Training finished.")
    return model

if __name__ == '__main__':
    # --- Configuration ---
    CSV_FILE = "toy_tts_parallel_data.csv"
    BATCH_SIZE = 32 # You can use a larger batch size on a GPU
    EPOCHS = 200

    # --- Set Device (GPU or CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # <-- NEW

    # --- Data Loading ---
    dataset = TextMelDataset(CSV_FILE)
    collate_fn = TextMelCollate()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # --- Model Initialization ---
    N_VOCAB = len(dataset.vocab)
    model = Tacotron2(n_vocab=N_VOCAB)
    model.to(device) # <-- Move the model to the GPU!

    # --- Training ---
    trained_model = train(model, dataloader, device, epochs=EPOCHS) # <-- Pass device to function

    # --- Save the Model ---
    torch.save(trained_model.state_dict(), 'tts_model.pth')
    print(f"Model saved to tts_model.pth. Trained on {device}.")