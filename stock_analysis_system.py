import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score
import tensorflow as tf
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import webbrowser

class StockAnalysisSystem:
    def __init__(self):
        self.setup_gui()
        
    def setup_gui(self):
        """Initialize the GUI"""
        self.root = tk.Tk()
        self.root.title("Stock Analysis System")
        self.root.geometry("1200x800")
        
        # Create main frames
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(fill='x', padx=5, pady=5)
        
        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add controls
        ttk.Label(self.control_frame, text="Stock Symbol:").pack(side='left', padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        self.symbol_entry = ttk.Entry(self.control_frame, textvariable=self.symbol_var)
        self.symbol_entry.pack(side='left', padx=5)
        
        ttk.Button(self.control_frame, text="Analyze", command=self.analyze_stock).pack(side='left', padx=5)
        
        # Add data source label with link
        source_label = ttk.Label(
            self.control_frame,
            text="Data Source: Yahoo Finance (finance.yahoo.com)",
            foreground='blue',
            cursor='hand2'
        )
        source_label.pack(side='right', padx=10)
        source_label.bind('<Button-1>', lambda e: self.open_yahoo_finance())
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.display_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.viz_tab = ttk.Frame(self.notebook)
        self.pred_tab = ttk.Frame(self.notebook)
        self.news_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data Analysis")
        self.notebook.add(self.viz_tab, text="Visualization")
        self.notebook.add(self.pred_tab, text="Predictions")
        self.notebook.add(self.news_tab, text="News")
        
        # Setup result displays
        self.setup_data_display()
        self.setup_viz_display()
        self.setup_pred_display()
        self.setup_news_display()
    
    def setup_data_display(self):
        """Setup the data analysis display"""
        self.data_text = tk.Text(self.data_tab, height=20, width=80)
        self.data_text.pack(padx=5, pady=5)
    
    def setup_viz_display(self):
        """Setup the visualization display"""
        self.fig = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_tab)
        self.canvas.get_tk_widget().pack()
    
    def setup_pred_display(self):
        """Setup the predictions display"""
        self.pred_text = tk.Text(self.pred_tab, height=20, width=80)
        self.pred_text.pack(padx=5, pady=5)
    
    def setup_news_display(self):
        """Setup the news display"""
        # Create a frame for news
        news_frame = ttk.Frame(self.news_tab)
        news_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(news_frame)
        scrollbar.pack(side='right', fill='y')
        
        # Configure text widget with scrollbar
        self.news_text = tk.Text(news_frame, height=30, width=80, wrap=tk.WORD,
                                yscrollcommand=scrollbar.set)
        self.news_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.news_text.yview)
        
        # Make links clickable
        self.news_text.tag_configure("link", foreground="blue", underline=1)
        self.news_text.tag_bind("link", "<Button-1>", self.open_news_link)
        
        # Configure other tags
        self.news_text.tag_configure("heading", font=("Helvetica", 12, "bold"))
        self.news_text.tag_configure("title", font=("Helvetica", 10, "bold"))
        self.news_text.tag_configure("info", font=("Helvetica", 9, "italic"))
    
    def fetch_stock_data(self, symbol):
        """Fetch stock data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period="1y")
            return df
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch stock data: {str(e)}")
            return None
    
    def process_data(self, df):
        """Process and analyze the stock data"""
        if df is None or df.empty:
            return "No data available"
        
        # Basic statistics
        stats = {
            "Average Close": df['Close'].mean(),
            "Std Dev": df['Close'].std(),
            "Min Price": df['Close'].min(),
            "Max Price": df['Close'].max(),
            "Trading Days": len(df),
            "Volume Mean": df['Volume'].mean()
        }
        
        # Calculate daily returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        return df, stats
    
    def visualize_data(self, df):
        """Create visualizations of the stock data"""
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        
        # Plot price and moving averages
        ax1.plot(df.index, df['Close'], label='Close Price')
        ax1.plot(df.index, df['MA20'], label='20-day MA')
        ax1.plot(df.index, df['MA50'], label='50-day MA')
        ax1.set_title('Stock Price and Moving Averages')
        ax1.legend()
        
        # Plot returns distribution
        sns.histplot(data=df['Returns'].dropna(), ax=ax2)
        ax2.set_title('Returns Distribution')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def predict_prices(self, df):
        """Implement machine learning predictions"""
        # Prepare features
        df['Target'] = df['Close'].shift(-1)
        df['Returns'] = df['Close'].pct_change()
        df['MA20_Ratio'] = df['Close'] / df['MA20']
        df['MA50_Ratio'] = df['Close'] / df['MA50']
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare features and target
        features = ['Close', 'Volume', 'Returns', 'MA20_Ratio', 'MA50_Ratio']
        X = df[features]
        y = df['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Prepare latest prediction
        latest_features = scaler.transform(X.iloc[[-1]])
        next_day_pred = model.predict(latest_features)[0]
        
        return {
            'MSE': mse,
            'R2': r2,
            'Next Day Prediction': next_day_pred,
            'Current Price': df['Close'].iloc[-1]
        }
    
    def fetch_news(self, symbol):
        """Fetch latest news about the stock"""
        try:
            # Get stock info
            stock = yf.Ticker(symbol)
            
            # Fetch news with more details
            all_news = stock.news
            
            # Process and format news items
            formatted_news = []
            for item in all_news[:10]:  # Get latest 10 news items
                news_item = {
                    'title': item.get('title', 'No title'),
                    'publisher': item.get('publisher', 'Unknown'),
                    'link': item.get('link', ''),
                    'published': datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d %H:%M'),
                    'summary': item.get('summary', 'No summary available')
                }
                formatted_news.append(news_item)
            
            return formatted_news
        except Exception as e:
            return [{'title': f"Error fetching news: {str(e)}"}]
    
    def update_news_display(self, news):
        """Update news display with formatted content"""
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, f"Latest News for {self.symbol_var.get().upper()}\n\n",
                            "heading")
        
        if not news:
            self.news_text.insert(tk.END, "No news available at this time.")
            return
        
        for item in news:
            # Add title
            self.news_text.insert(tk.END, f"📰 {item['title']}\n", "title")
            
            # Add publisher and date
            self.news_text.insert(tk.END, 
                                f"Published by {item['publisher']} on {item['published']}\n",
                                "info")
            
            # Add summary
            self.news_text.insert(tk.END, f"\n{item['summary']}\n")
            
            # Add link
            self.news_text.insert(tk.END, "Read more: ")
            self.news_text.insert(tk.END, f"{item['link']}\n", "link")
            
            # Add separator
            self.news_text.insert(tk.END, "\n" + "-"*80 + "\n\n")
    
    def analyze_stock(self):
        """Main analysis function"""
        symbol = self.symbol_var.get().upper()
        
        # Show loading message
        self.news_text.delete(1.0, tk.END)
        self.news_text.insert(tk.END, "Fetching latest news...\n")
        
        # Fetch and process data
        df = self.fetch_stock_data(symbol)
        if df is not None:
            df, stats = self.process_data(df)
            
            # Update data analysis tab
            self.data_text.delete(1.0, tk.END)
            self.data_text.insert(tk.END, "Stock Analysis Results\n\n")
            for key, value in stats.items():
                self.data_text.insert(tk.END, f"{key}: {value:.2f}\n")
            
            # Update visualization tab
            self.visualize_data(df)
            
            # Update predictions tab
            pred_results = self.predict_prices(df)
            self.pred_text.delete(1.0, tk.END)
            self.pred_text.insert(tk.END, "Prediction Results\n\n")
            for key, value in pred_results.items():
                self.pred_text.insert(tk.END, f"{key}: {value:.2f}\n")
            
            # Update news tab with enhanced display
            news = self.fetch_news(symbol)
            self.update_news_display(news)
    
    def open_news_link(self, event):
        """Open news link in browser"""
        tag_indices = self.news_text.tag_prevrange(
            "link",
            self.news_text.index("@%s,%s" % (event.x, event.y))
        )
        if tag_indices:
            link = self.news_text.get(*tag_indices)
            webbrowser.open(link)
    
    def open_yahoo_finance(self):
        """Open Yahoo Finance website"""
        symbol = self.symbol_var.get().upper()
        url = f"https://finance.yahoo.com/quote/{symbol}"
        webbrowser.open(url)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    app = StockAnalysisSystem()
    app.run()

if __name__ == "__main__":
    main() 