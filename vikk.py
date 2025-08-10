import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import spacy
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Prediction Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load spaCy model (you'll need to install: python -m spacy download en_core_web_sm)
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
        return None

nlp = load_nlp_model()

class StockPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        
    def prepare_features(self, df):
        """Prepare technical indicators as features"""
        # Calculate technical indicators
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        # Lag features
        for i in [1, 2, 3, 5]:
            df[f'Close_lag_{i}'] = df['Close'].shift(i)
            
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_model(self, df, model_type='RandomForest'):
        """Train the prediction model"""
        df = self.prepare_features(df)
        df = df.dropna()
        
        # Features for training
        feature_cols = ['SMA_10', 'SMA_30', 'RSI', 'MACD', 'Volatility', 
                       'Price_Change', 'Volume_MA', 'Close_lag_1', 'Close_lag_2', 
                       'Close_lag_3', 'Close_lag_5']
        
        X = df[feature_cols]
        y = df['Close']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'RandomForest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            self.model = LinearRegression()
            
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse,
            'test_actual': y_test,
            'test_pred': test_pred,
            'feature_cols': feature_cols
        }
    
    def predict_future(self, df, days=5):
        """Predict future prices"""
        df = self.prepare_features(df)
        feature_cols = ['SMA_10', 'SMA_30', 'RSI', 'MACD', 'Volatility', 
                       'Price_Change', 'Volume_MA', 'Close_lag_1', 'Close_lag_2', 
                       'Close_lag_3', 'Close_lag_5']
        
        predictions = []
        current_data = df.iloc[-1:].copy()
        
        for _ in range(days):
            X = current_data[feature_cols].values
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update data for next prediction (simplified approach)
            current_data = current_data.copy()
            current_data.loc[current_data.index[0], 'Close'] = pred
            current_data = self.prepare_features(current_data)
            
        return predictions

class NLPProcessor:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.stock_keywords = ['stock', 'share', 'equity', 'ticker', 'company']
        self.prediction_keywords = ['predict', 'forecast', 'future', 'tomorrow', 'next']
        self.analysis_keywords = ['analyze', 'analysis', 'trend', 'chart', 'graph']
    
    def extract_stock_symbol(self, text):
        """Extract stock symbol from text"""
        # Look for patterns like $AAPL, AAPL, Apple Inc
        patterns = [
            r'\$([A-Z]{1,5})',  # $AAPL
            r'\b([A-Z]{2,5})\b',  # AAPL
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                return matches[0]
        
        # Use NLP to extract company names
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON']:
                    # Try common stock symbols
                    symbol_map = {
                        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
                        'amazon': 'AMZN', 'tesla': 'TSLA', 'meta': 'META',
                        'nvidia': 'NVDA', 'netflix': 'NFLX'
                    }
                    entity_text = ent.text.lower()
                    for company, symbol in symbol_map.items():
                        if company in entity_text:
                            return symbol
        
        return None
    
    def extract_intent(self, text):
        """Extract user intent from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.prediction_keywords):
            return 'predict'
        elif any(word in text_lower for word in self.analysis_keywords):
            return 'analyze'
        else:
            return 'general'
    
    def extract_timeframe(self, text):
        """Extract timeframe from text"""
        text_lower = text.lower()
        
        if 'day' in text_lower:
            days_match = re.search(r'(\d+)\s*days?', text_lower)
            if days_match:
                return int(days_match.group(1))
            return 1
        elif 'week' in text_lower:
            weeks_match = re.search(r'(\d+)\s*weeks?', text_lower)
            if weeks_match:
                return int(weeks_match.group(1)) * 7
            return 7
        elif 'month' in text_lower:
            return 30
        
        return 5  # default

def create_stock_chart(df, predictions=None, stock_symbol=""):
    """Create interactive stock chart"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=f'{stock_symbol} Price'
    ))
    
    # Add predictions if provided
    if predictions:
        future_dates = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=len(predictions),
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predictions',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title=f'{stock_symbol} Stock Price Analysis',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600
    )
    
    return fig

def create_volume_chart(df, stock_symbol=""):
    """Create volume chart"""
    fig = px.bar(df, x=df.index, y='Volume', title=f'{stock_symbol} Trading Volume')
    fig.update_layout(height=300)
    return fig

def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'div_yield': info.get('dividendYield', 0)
        }
    except:
        return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()

# Sidebar
st.sidebar.title("Stock Prediction Chatbot 📈")
st.sidebar.markdown("---")

# Sample questions
st.sidebar.subheader("Try these examples:")
sample_questions = [
    "Predict AAPL for next 5 days",
    "Analyze Tesla stock trend",
    "What's the forecast for Microsoft?",
    "Show me Google stock analysis",
    "$NVDA prediction next week"
]

for question in sample_questions:
    if st.sidebar.button(question):
        st.session_state.user_input = question

# Main interface
st.title("🤖 AI Stock Prediction Chatbot")
st.markdown("Ask me about stock predictions, analysis, or trends using natural language!")

# Chat interface
chat_container = st.container()

# Input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input(
    "Ask me about stocks:",
    value=st.session_state.user_input,
    placeholder="e.g., 'Predict Apple stock for next week' or 'Analyze Tesla trend'"
)

if st.button("Send") or user_input:
    if user_input:
        # Process user input with NLP
        if nlp:
            nlp_processor = NLPProcessor(nlp)
            
            # Extract information
            stock_symbol = nlp_processor.extract_stock_symbol(user_input)
            intent = nlp_processor.extract_intent(user_input)
            timeframe = nlp_processor.extract_timeframe(user_input)
            
            # Add to chat history
            st.session_state.chat_history.append(("user", user_input))
            
            if stock_symbol:
                try:
                    # Get stock data
                    with st.spinner(f"Fetching {stock_symbol} data..."):
                        stock = yf.Ticker(stock_symbol)
                        df = stock.history(period="1y")
                        
                        if df.empty:
                            st.error(f"Could not find data for {stock_symbol}")
                        else:
                            # Get stock info
                            stock_info = get_stock_info(stock_symbol)
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                if intent == 'predict':
                                    st.subheader(f"📊 Prediction for {stock_symbol}")
                                    
                                    # Train model and make predictions
                                    with st.spinner("Training prediction model..."):
                                        results = st.session_state.predictor.train_model(df.copy())
                                        predictions = st.session_state.predictor.predict_future(df.copy(), timeframe)
                                    
                                    # Show model performance
                                    st.metric("Model R² Score", f"{results['test_r2']:.3f}")
                                    st.metric("RMSE", f"${results['rmse']:.2f}")
                                    
                                    # Show predictions
                                    current_price = df['Close'].iloc[-1]
                                    predicted_price = predictions[-1]
                                    change = ((predicted_price - current_price) / current_price) * 100
                                    
                                    st.metric(
                                        f"Predicted Price ({timeframe} days)",
                                        f"${predicted_price:.2f}",
                                        f"{change:+.2f}%"
                                    )
                                    
                                    # Create chart with predictions
                                    fig = create_stock_chart(df, predictions, stock_symbol)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show prediction table
                                    future_dates = pd.date_range(
                                        start=df.index[-1] + timedelta(days=1),
                                        periods=len(predictions),
                                        freq='D'
                                    )
                                    pred_df = pd.DataFrame({
                                        'Date': future_dates,
                                        'Predicted Price': [f"${p:.2f}" for p in predictions]
                                    })
                                    st.dataframe(pred_df, use_container_width=True)
                                    
                                    response = f"I've analyzed {stock_symbol} and created a {timeframe}-day prediction. The model shows an R² score of {results['test_r2']:.3f}."
                                    
                                else:  # analyze
                                    st.subheader(f"📈 Analysis for {stock_symbol}")
                                    
                                    # Show current metrics
                                    current_price = df['Close'].iloc[-1]
                                    prev_price = df['Close'].iloc[-2]
                                    change = current_price - prev_price
                                    change_pct = (change / prev_price) * 100
                                    
                                    col1a, col1b, col1c = st.columns(3)
                                    with col1a:
                                        st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}")
                                    with col1b:
                                        st.metric("Daily Change", f"{change_pct:+.2f}%")
                                    with col1c:
                                        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
                                    
                                    # Create analysis chart
                                    fig = create_stock_chart(df, None, stock_symbol)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Volume chart
                                    vol_fig = create_volume_chart(df, stock_symbol)
                                    st.plotly_chart(vol_fig, use_container_width=True)
                                    
                                    response = f"Here's the analysis for {stock_symbol}. Current price is ${current_price:.2f} with a daily change of {change_pct:+.2f}%."
                            
                            with col2:
                                if stock_info:
                                    st.subheader("Company Info")
                                    st.write(f"**Name:** {stock_info['name']}")
                                    st.write(f"**Sector:** {stock_info['sector']}")
                                    if stock_info['market_cap']:
                                        st.write(f"**Market Cap:** ${stock_info['market_cap']:,.0f}")
                                    if stock_info['pe_ratio']:
                                        st.write(f"**P/E Ratio:** {stock_info['pe_ratio']:.2f}")
                                    if stock_info['div_yield']:
                                        st.write(f"**Dividend Yield:** {stock_info['div_yield']*100:.2f}%")
                                
                                # Recent performance
                                st.subheader("Recent Performance")
                                returns_1d = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100
                                returns_1w = ((df['Close'].iloc[-1] / df['Close'].iloc[-8]) - 1) * 100
                                returns_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-21]) - 1) * 100
                                
                                st.metric("1 Day", f"{returns_1d:+.2f}%")
                                st.metric("1 Week", f"{returns_1w:+.2f}%")
                                st.metric("1 Month", f"{returns_1m:+.2f}%")
                        
                        st.session_state.chat_history.append(("assistant", response))
                        
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error processing {stock_symbol}: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))
            
            else:
                help_msg = """I couldn't identify a stock symbol in your message. Please try:
                - Using a stock symbol like AAPL, TSLA, or MSFT
                - Mentioning a company name like Apple, Tesla, or Microsoft  
                - Adding a $ before the symbol like $AAPL
                
                Examples: "Predict AAPL for 5 days" or "Analyze Tesla stock trend"
                """
                st.warning(help_msg)
                st.session_state.chat_history.append(("assistant", help_msg))
        
        else:
            st.error("NLP model not loaded. Please install spaCy English model.")
    
    # Clear input
    st.session_state.user_input = ""
    st.experimental_rerun()

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.subheader("Chat History")
    
    for role, message in st.session_state.chat_history[-10:]:  # Show last 10 messages
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Assistant:** {message}")

# Footer
st.markdown("---")
st.markdown("*Powered by yfinance, spaCy, scikit-learn, and Streamlit*")
st.caption("⚠️ This is for educational purposes only. Not financial advice!")