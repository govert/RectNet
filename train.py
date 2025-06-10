import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from detector import model

# --- Create the training DataLoader ---
train_images = torch.load('data/train_images.pt')
train_labels = torch.load('data/train_labels.pt')
train_dataset = TensorDataset(train_images, train_labels)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
num_epochs = 1
for epoch in range(num_epochs):
    model.train() # Set the model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print statistics
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")

print("\nFinished Training! 🎉")

# --- SAVE THE TRAINED MODEL ---
MODEL_PATH = 'rectnet_model.pth'
torch.save(model.state_dict(), MODEL_PATH)

print(f"Model saved to {MODEL_PATH}")