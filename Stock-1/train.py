import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import os
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# File paths
MODEL_PATH = "stock_model.pth"
DATA_PATH = "stock_training_data.json"

# Define the Stock Predictor Model
class StockPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Fetch Stock Data with Technical Indicators
def fetch_stock_data(ticker, period='1y', interval='1d'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)

    if hist.empty:
        return None

    # Compute Moving Averages
    hist['SMA_10'] = hist['Close'].rolling(window=10).mean()
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()

    # Compute Relative Strength Index (RSI)
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    hist['RSI'] = 100 - (100 / (1 + rs))

    # Compute Moving Average Convergence Divergence (MACD)
    ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
    ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
    hist['MACD'] = ema12 - ema26

    # Compute Bollinger Bands
    hist['BB_upper'] = hist['Close'].rolling(20).mean() + (hist['Close'].rolling(20).std() * 2)
    hist['BB_lower'] = hist['Close'].rolling(20).mean() - (hist['Close'].rolling(20).std() * 2)

    hist.dropna(inplace=True)  # Remove NaN values

    return hist[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'RSI', 'MACD']].values

# Prepare Data for Training
def prepare_data(data, sequence_length=10):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length, :])  # Features
        y.append(data[i + sequence_length, 3])  # Predict Close price

    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1), scaler

# Save Training Data
def save_training_data(ticker, actual_price, predicted_price):
    data = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            data = json.load(f)

    # Add new entry
    data.append({
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "actual_price": actual_price,
        "predicted_price": predicted_price
    })

    # Keep only the last 500 records
    data = data[-500:]

    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=4)

# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch):
    if epoch % 20 == 0 and epoch != 0:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.8  # Reduce learning rate
            print(f"🔽 Adjusted Learning Rate: {param_group['lr']:.6f}")

# Train the AI Model
def train_model(ticker='AAPL', epochs=100):
    print(f"📊 Training model for {ticker}...")

    data = fetch_stock_data(ticker)
    if data is None:
        print("⚠️ No stock data available.")
        return
    
    X_train, y_train, scaler = prepare_data(data)

    model = StockPredictor()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print("✅ Loaded existing model weights.")
        except Exception as e:
            print("⚠️ Model architecture changed! Training from scratch.")

        print("✅ Loaded existing model for retraining.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer, epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model trained and saved for {ticker}.")

# Load the Model for Predictions
def load_model():
    model = StockPredictor()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print("✅ Model loaded successfully.")
    else:
        print("⚠️ No trained model found. Please train first.")
    return model

# Predict Next Day's Stock Price
def predict_price(ticker):
    model = load_model()
    model.eval()

    data = fetch_stock_data(ticker)
    if data is None:
        return "⚠️ No stock data available."

    X_test, y_actual, scaler = prepare_data(data)

    with torch.no_grad():
        predicted_price = model(X_test[-1].unsqueeze(0)).item()
    
    actual_price = y_actual[-1].item()
    predicted_price = scaler.inverse_transform([[0, 0, 0, predicted_price, 0, 0, 0, 0]])[0][3]  # Reverse scale

    save_training_data(ticker, actual_price, predicted_price)

    print(f"🔹 Actual: ${actual_price:.2f} | Predicted: ${predicted_price:.2f}")

    return predicted_price

# Continuous Backtesting Mode
def continuous_backtesting():
    print("🔄 Running Continuous Backtesting Mode...")
    if not os.path.exists(DATA_PATH):
        print("⚠️ No historical data for backtesting.")
        return

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    errors = []
    for entry in data[-50:]:  # Analyze last 50 predictions
        actual = entry["actual_price"]
        predicted = entry["predicted_price"]
        error = abs(actual - predicted) / actual
        errors.append(error)

    avg_error = (sum(errors) / len(errors)) * 100
    print(f"📊 Backtest Error: {avg_error:.2f}%")

    if avg_error > 5.0:
        print("⚠️ High AI error! Retraining...")
        train_model()

# Investment Strategy Decision
def investment_strategy(ticker, mode):
    price = predict_price(ticker)

    if mode == "long_term":
        print("📈 Long-Term Strategy: Evaluating trends and moving averages...")
        if price > 200:  # Example threshold
            return "BUY for long-term"
        else:
            return "HOLD"

    elif mode == "day_trading":
        print("📊 Day Trading Mode: Looking at short-term trends...")
        if price > 200:
            return "BUY NOW"
        elif price < 180:
            return "SELL NOW"
        else:
            return "WAIT"

# AI Self-Improvement
def ai_self_improvement():
    print("🔄 AI is reviewing past predictions to improve accuracy...")
    continuous_backtesting()

if __name__ == "__main__":
    ticker = 'AAPL'
    train_model(ticker, epochs=100)
    predict_price(ticker)
    ai_self_improvement()
def adaptive_train(mistakes):
    if not mistakes:
        print("✅ No mistakes to correct.")
        return
    
    print(f"🔄 Adapting model with {len(mistakes)} mistakes...")
    
    model = load_model()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Prepare mistake data
    x_train = [m[1] for m in mistakes]
    y_train = [m[2] for m in mistakes]

    x_train = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model.train()
    for epoch in range(10):  # Only fine-tune for 10 epochs
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(x_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Fine-Tune Loss = {total_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Adaptive learning complete! Model updated.")
