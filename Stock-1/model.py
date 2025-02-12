import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from data_fetch import get_stock_data, get_news_sentiment  # Ensure this module provides stock prices & sentiment

MODEL_PATH = "stock_model.pth"
WINDOW_SIZE = 20  # Use a window size of 20 for the training data

class GRUStockPredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, dropout=0.2):
        """
        GRU-based stock price prediction model.
        :param input_size: Number of features (price, volume, sentiment)
        :param hidden_size: Size of hidden layer
        :param num_layers: Number of GRU layers
        :param dropout: Dropout rate to prevent overfitting
        """
        super(GRUStockPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass for the GRU model.
        :param x: Input tensor of shape (batch_size, seq_length, input_size)
        :return: Predicted stock price for each sample
        """
        gru_out, _ = self.gru(x)  # GRU returns the output for all time steps
        return self.fc(gru_out[:, -1, :])  # Use only the last time step output for prediction

def prepare_training_data(stock_prices, volumes, sentiments, window_size=WINDOW_SIZE):
    """
    Prepare training data by stacking stock prices, volumes, and sentiment scores.
    Normalizes each feature to a range between 0 and 1.
    :param stock_prices: Historical stock prices
    :param volumes: Trading volumes
    :param sentiments: Sentiment scores for the stock
    :param window_size: Number of previous data points to use in each sample
    :return: Prepared input and output tensors for training
    """
    stock_prices = np.array(stock_prices, dtype=np.float32)
    volumes = np.array(volumes, dtype=np.float32)
    sentiments = np.array(sentiments, dtype=np.float32)

    # Normalize stock prices and volumes using MinMaxScaler to scale between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    stock_prices = scaler.fit_transform(stock_prices.reshape(-1, 1)).flatten()
    volumes = scaler.fit_transform(volumes.reshape(-1, 1)).flatten()
    
    # Normalize sentiments to a range between 0 and 1
    sentiments = (sentiments - np.min(sentiments)) / (np.max(sentiments) - np.min(sentiments) + 1e-7)

    X, y = [], []
    for i in range(len(stock_prices) - window_size):
        # Ensure the data shape is (batch_size, seq_length, input_size), where input_size = 3
        X.append(np.column_stack((
            stock_prices[i:i + window_size], 
            volumes[i:i + window_size], 
            sentiments[i:i + window_size]
        )))
        y.append(stock_prices[i + window_size])  # Predict the next stock price
    
    # Convert to torch tensors
    X = np.array(X)  # Convert list of arrays to numpy array
    y = np.array(y)
    
    # Ensure X has the shape (batch_size, seq_length, input_size)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

def train_model(ticker, epochs=100, lr=0.001):
    """
    Train the GRU-based stock prediction model.
    :param ticker: The stock ticker symbol to fetch data for
    :param epochs: Number of training epochs
    :param lr: Learning rate for the optimizer
    """
    stock_prices, volumes = get_stock_data(ticker, period='1y', include_volume=True)
    sentiments = get_news_sentiment(ticker)

    if len(stock_prices) < WINDOW_SIZE:
        raise ValueError("Not enough data to train the model.")

    X_train, y_train = prepare_training_data(stock_prices, volumes, sentiments)

    print(f"X_train shape: {X_train.shape}")  # Check the shape of the data before training

    model = GRUStockPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Model Training Complete & Saved!")


def load_model():
    """
    Load the pre-trained stock prediction model.
    :return: The model with the trained parameters
    """
    model = GRUStockPredictor()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # Set model to evaluation mode
    else:
        print("Warning: No pre-trained model found. Training a new model.")
    return model
