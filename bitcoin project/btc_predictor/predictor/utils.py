# predictor/utils.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from .models import BitcoinData

def process_csv_data(csv_file):
    """Process uploaded CSV data and save to database"""
    df = pd.read_csv(csv_file)
    
    # Clear existing data
    BitcoinData.objects.all().delete()
    
    # Process and save data
    for _, row in df.iterrows():
        try:
            bitcoin_data = BitcoinData(
                date=pd.to_datetime(row['Date']).date(),
                open_price=float(row['Open']),
                high_price=float(row['High']),
                low_price=float(row['Low']),
                close_price=float(row['Close']),
                volume=float(row['Volume'])
            )
            bitcoin_data.save()
        except Exception as e:
            print(f"Error processing row: {e}")
    
    return len(df)

def get_historical_data():
    """Get historical Bitcoin data from database"""
    data = BitcoinData.objects.all().order_by('date')
    
    if not data:
        return None
        
    df = pd.DataFrame(list(data.values()))
    
    # Rename columns to match original analysis
    column_mapping = {
        'date': 'Date',
        'open_price': 'Open',
        'high_price': 'High',
        'low_price': 'Low',
        'close_price': 'Close',
        'volume': 'Volume'
    }
    df = df.rename(columns=column_mapping)
    
    return df

def prepare_features(df):
    """Prepare features for model training"""
    if df is None or df.empty:
        return None, None
    
    # Convert date to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Extract date components
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    
    # Create features
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    
    # Drop the row with NaN target (last row)
    df = df.dropna(subset=['target'])
    
    # Prepare features and target
    features = df[['open-close', 'low-high', 'is_quarter_end']]
    target = df['target']
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, target

def train_model(X, y, model_type):
    """Train selected model"""
    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'svc':
        model = SVC(kernel='poly', probability=True)
    else:  # xgboost
        model = XGBClassifier()
    
    # Train on first 70% of data
    split_idx = int(len(X) * 0.7)
    X_train, X_valid = X[:split_idx], X[split_idx:]
    y_train, y_valid = y[:split_idx], y[split_idx:]
    
    model.fit(X_train, y_train)
    
    # Get predictions and metrics
    train_proba = model.predict_proba(X_train)[:, 1]
    valid_proba = model.predict_proba(X_valid)[:, 1]
    
    train_auc = metrics.roc_auc_score(y_train, train_proba)
    valid_auc = metrics.roc_auc_score(y_valid, valid_proba)
    
    # Get last prediction (for tomorrow)
    last_features = X[-1].reshape(1, -1)
    prediction = model.predict(last_features)[0]
    confidence = model.predict_proba(last_features)[0][1] * 100
    
    # Calculate confusion matrix
    y_pred = model.predict(X_valid)
    cm = metrics.confusion_matrix(y_valid, y_pred)
    
    return {
        'model': model,
        'prediction': bool(prediction),
        'confidence': confidence if prediction else 100 - confidence,
        'train_auc': train_auc,
        'valid_auc': valid_auc,
        'confusion_matrix': cm,
        'X_valid': X_valid,
        'y_valid': y_valid
    }

def get_price_chart(df):
    """Generate Bitcoin price chart"""
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.title('Bitcoin Close Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Convert to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic

def get_confusion_matrix_plot(cm):
    """Generate confusion matrix visualization"""
    plt.figure(figsize=(6, 4))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Convert to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    graphic = base64.b64encode(image_png).decode('utf-8')
    return graphic