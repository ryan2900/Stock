import yfinance as yf
import requests
from newspaper import Article
from bs4 import BeautifulSoup
import numpy as np
import random
import datetime
import re

def get_stock_data(ticker, period="6mo", include_volume=False):
    """
    Fetch historical stock data for a given ticker symbol.
    
    :param ticker: The stock symbol (e.g., 'AAPL').
    :param period: The period to fetch data for (default: '6mo').
    :param include_volume: Whether to include trading volume data (default: False).
    :return: Stock data (close prices and optionally volumes).
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            raise ValueError(f"No stock data found for {ticker}.")
        
        if include_volume:
            return hist['Close'].values, hist['Volume'].values
        else:
            return hist['Close'].values
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return np.array([]), np.array([])  # Return empty arrays on error


def get_news_headlines(ticker, num_headlines=5):
    """
    Fetch the latest news headlines for a given stock ticker.

    :param ticker: The stock symbol (e.g., 'AAPL').
    :param num_headlines: Number of headlines to return (default: 5).
    :return: A list of news headlines.
    """
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        headlines = []
        for g in soup.find_all("h3"):
            if g.text:
                headlines.append(g.text)

        return headlines[:num_headlines]  # Return the top 'num_headlines' headlines
    except Exception as e:
        print(f"Error fetching news headlines: {e}")
        return []


def get_news_sentiment(ticker, period="6mo"):
    """
    Extract sentiment scores based on news articles related to a stock ticker.

    :param ticker: The stock symbol (e.g., 'AAPL').
    :param period: The period to fetch news sentiment for (default: '6mo').
    :return: The average sentiment score based on the news.
    """
    try:
        headlines = get_news_headlines(ticker)
        sentiment_scores = []

        for headline in headlines:
            # Search for an actual news article using Google
            search_url = f"https://www.google.com/search?q={headline.replace(' ', '+')}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            for a in soup.find_all("a", href=True):
                if "url?q=" in a["href"] and "google.com" not in a["href"]:
                    match = re.search(r"url\?q=(https?://\S+)&", a["href"])
                    if match:
                        links.append(match.group(1))

            # Process only the first valid link
            if links:
                article = Article(links[0])
                article.download()
                article.parse()
                article.nlp()
                # Simple sentiment extraction from article summary (positive or negative counts)
                sentiment_scores.append(article.summary.count("positive") - article.summary.count("negative"))

        # Calculate the average sentiment score
        if sentiment_scores:
            return np.mean(sentiment_scores)
        return random.uniform(-1, 1)  # Return a random sentiment score if no data is available

    except Exception as e:
        print(f"Error extracting news sentiment: {e}")
        return random.uniform(-1, 1)  # Return a random score as a fallback


def get_economic_indicators(period="6mo"):
    """
    Fetch macroeconomic indicators (e.g., interest rates, inflation data).
    
    :param period: The period for which to fetch economic data (default: '6mo').
    :return: A dictionary of relevant economic indicators (mock data for now).
    """
    try:
        # Placeholder for fetching actual economic indicators. For now, using mock values.
        indicators = {
            "interest_rate": 0.05,  # Example interest rate (5%)
            "inflation_rate": 0.02,  # Example inflation rate (2%)
        }
        return indicators
    except Exception as e:
        print(f"Error fetching economic indicators: {e}")
        return {}


def fetch_all_data(ticker, period="6mo"):
    """
    Fetch all relevant data (stock prices, news sentiment, economic indicators) for a given ticker.

    :param ticker: The stock symbol (e.g., 'AAPL').
    :param period: The period to fetch data for (default: '6mo').
    :return: A tuple containing stock data, sentiment, and economic indicators.
    """
    stock_prices, volumes = get_stock_data(ticker, period, include_volume=True)
    sentiment = get_news_sentiment(ticker, period)
    economic_indicators = get_economic_indicators(period)

    return stock_prices, volumes, sentiment, economic_indicators
