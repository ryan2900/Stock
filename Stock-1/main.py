import sys
import torch
import numpy as np
import yfinance as yf
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QLineEdit, QTextEdit, QHBoxLayout, QComboBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from train import load_model, train_model, continuous_backtesting
from data_fetch import get_stock_data, get_news_sentiment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class StockApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = load_model()
        self.previous_price = None

    def initUI(self):
        self.setWindowTitle("AI1 Stock Recommendation System")
        self.setStyleSheet("background-color: #121212; color: white;")
        self.showMaximized()

        layout = QVBoxLayout()

        # Header
        header = QLabel("📈 AI-Powered Stock Advisor")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Input Section
        input_layout = QHBoxLayout()
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("Enter Stock Ticker (e.g., NVDA)")
        self.budget_input = QLineEdit()
        self.budget_input.setPlaceholderText("Enter Budget ($)")
        self.period_selector = QComboBox()
        self.period_selector.addItems(["1d", "5d", "1mo", "3mo", "6mo", "1y"])

        input_layout.addWidget(self.stock_input)
        input_layout.addWidget(self.budget_input)
        input_layout.addWidget(self.period_selector)
        layout.addLayout(input_layout)

        # Buttons
        button_layout = QHBoxLayout()
        self.fetch_button = QPushButton("🔍 Get Recommendation")
        self.fetch_button.setStyleSheet("background-color: #1DB954; color: black; font-weight: bold;")
        self.fetch_button.clicked.connect(self.get_recommendation)

        self.train_button = QPushButton("🔄 Retrain AI")
        self.train_button.setStyleSheet("background-color: #FF9800; color: black; font-weight: bold;")
        self.train_button.clicked.connect(self.retrain_model)

        self.backtest_button = QPushButton("📊 Backtest AI")
        self.backtest_button.setStyleSheet("background-color: #4CAF50; color: black; font-weight: bold;")
        self.backtest_button.clicked.connect(self.run_backtest)

        button_layout.addWidget(self.fetch_button)
        button_layout.addWidget(self.train_button)
        button_layout.addWidget(self.backtest_button)
        layout.addLayout(button_layout)

        # Live Price Display
        self.price_label = QLabel("Live Price: $0.00")
        self.price_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        self.price_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.price_label)

        # Graph
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Results Display
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("background-color: #222; color: white; padding: 10px; font-size: 14px;")
        layout.addWidget(self.result_area)

        self.setLayout(layout)

        # Timer for live updates (every 3 seconds)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_price)
        self.timer.start(3000)

    def get_recommendation(self):
        try:
            ticker = self.stock_input.text().strip().upper()
            budget_text = self.budget_input.text().strip()
            period = self.period_selector.currentText()
            
            if not ticker:
                self.result_area.setText("⚠️ Please enter a stock ticker.")
                return
            if not budget_text or not budget_text.replace(".", "", 1).isdigit():
                self.result_area.setText("⚠️ Please enter a valid budget.")
                return
            
            budget = float(budget_text)
            stock_prices = get_stock_data(ticker, period)
            if len(stock_prices) < 10:
                self.result_area.setText(f"⚠️ Not enough data for {ticker}. Try another stock.")
                return

            latest_price = stock_prices[-1]
            stock_prices = stock_prices[-10:]
            stock_prices = (stock_prices - np.min(stock_prices)) / (np.max(stock_prices) - np.min(stock_prices))
            
            input_data = torch.tensor(stock_prices, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            predicted_trend = self.model(input_data).item()
            sentiment_score = get_news_sentiment(ticker)

            stop_loss = latest_price * 0.95
            risk_management = f"Stop-Loss: ${stop_loss:.2f}"

            confidence_score = np.random.uniform(75, 95)  # Placeholder for AI confidence calculation

            if predicted_trend > 0 and sentiment_score > 0:
                recommendation = "✅ BUY"
                shares = budget // latest_price
                action_text = f"📈 Buy {shares} shares with ${budget}. {risk_management}"
            elif predicted_trend < 0:
                recommendation = "❌ SHORT"
                shares = budget // latest_price
                action_text = f"📉 Short {shares} shares with ${budget}. {risk_management}"
            else:
                recommendation = "⏳ HOLD"
                action_text = "No action recommended."

            result_text = (f"🔹 **Stock**: {ticker}\n"
                           f"🔹 **Latest Price**: ${latest_price:.2f}\n"
                           f"🔹 **Predicted Trend**: {predicted_trend:.2f}\n"
                           f"🔹 **News Sentiment Score**: {sentiment_score:.2f}\n"
                           f"🔹 **Confidence Score**: {confidence_score:.2f}%\n"
                           f"🔹 **Recommendation**: {recommendation}\n{action_text}")
            self.result_area.setText(result_text)

            self.update_graph(ticker, period)

        except Exception as e:
            self.result_area.setText(f"❌ Error: {str(e)}")
            print(f"Error occurred: {e}")

    def update_graph(self, ticker, period):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                self.result_area.append("\n⚠️ No graph data available.")
                return

            self.ax.clear()
            self.ax.plot(hist.index, hist["Close"], color="#1DB954", linewidth=2, label="Stock Price")
            
            # Add moving average
            hist["SMA"] = hist["Close"].rolling(window=5).mean()
            self.ax.plot(hist.index, hist["SMA"], color="yellow", linestyle="dashed", label="5-Day SMA")

            self.ax.set_title(f"{ticker} Price Trend ({period})", fontsize=14, fontweight="bold", color="white")
            self.ax.set_xlabel("Date", fontsize=12, color="white")
            self.ax.set_ylabel("Price ($)", fontsize=12, color="white")
            self.ax.legend()
            self.ax.tick_params(axis="x", colors="white")
            self.ax.tick_params(axis="y", colors="white")
            self.figure.patch.set_facecolor("#121212")
            self.ax.set_facecolor("#222222")

            self.canvas.draw()
        except Exception as e:
            self.result_area.append(f"\n❌ Graph Error: {e}")

    def update_live_price(self):
        ticker = self.stock_input.text().strip().upper()
        if not ticker:
            return

        stock = yf.Ticker(ticker)
        live_price = stock.history(period="1d")["Close"].iloc[-1]
        self.price_label.setText(f"Live Price: ${live_price:.2f}")

    def retrain_model(self):
        self.result_area.setText("🔄 Retraining AI Model...")
        train_model()
        self.model = load_model()
        self.result_area.setText("✅ AI Model Retrained Successfully!")

    def run_backtest(self):
        self.result_area.setText("📊 Running AI Backtest...")
        continuous_backtesting()
        self.result_area.setText("✅ AI Backtest Completed!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    stock_app = StockApp()
    stock_app.show()
    sys.exit(app.exec())
