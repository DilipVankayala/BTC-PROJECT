from django.shortcuts import render

# Create your views here.

# predictor/views.py

from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import CSVUploadForm, PredictionForm
from .utils import (process_csv_data, get_historical_data, prepare_features,
                    train_model, get_price_chart, get_confusion_matrix_plot)
from .models import PredictionResult

def index(request):
    """Main page view"""
    # Check if we have data
    data_exists = False
    if get_historical_data() is not None:
        data_exists = True
    
    prediction_form = PredictionForm()
    csv_form = CSVUploadForm()
    
    context = {
        'prediction_form': prediction_form,
        'csv_form': csv_form,
        'data_exists': data_exists
    }
    
    return render(request, 'predictor/index.html', context)

def predict(request):
    """Generate prediction based on selected model"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            model_choice = form.cleaned_data['model_choice']
            
            # Get data and prepare features
            df = get_historical_data()
            if df is None:
                messages.error(request, 'No Bitcoin data available. Please upload CSV data first.')
                return redirect('index')
            
            features, target = prepare_features(df)
            if features is None:
                messages.error(request, 'Error preparing features.')
                return redirect('index')
            
            # Train model and get prediction
            results = train_model(features, target, model_choice)
            
            # Save prediction to database
            prediction_record = PredictionResult(
                prediction=results['prediction'],
                confidence=results['confidence'],
                model_used=model_choice
            )
            prediction_record.save()
            
            # Generate charts
            price_chart = get_price_chart(df)
            cm_chart = get_confusion_matrix_plot(results['confusion_matrix'])
            
            model_name_mapping = {
                'logistic': 'Logistic Regression',
                'svc': 'Support Vector Classifier',
                'xgboost': 'XGBoost Classifier'
            }
            
            context = {
                'prediction': results['prediction'],
                'confidence': results['confidence'],
                'model_name': model_name_mapping[model_choice],
                'train_auc': results['train_auc'] * 100,
                'valid_auc': results['valid_auc'] * 100,
                'price_chart': price_chart,
                'cm_chart': cm_chart,
                'last_date': df['Date'].iloc[-1]
            }
            
            return render(request, 'predictor/result.html', context)
    
    return redirect('index')

def upload_csv(request):
    """Handle CSV file upload"""
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            
            # Check if it's a CSV file
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'File is not a CSV')
                return redirect('index')
            
            # Process the file
            try:
                rows_processed = process_csv_data(csv_file)
                messages.success(request, f'Successfully uploaded and processed {rows_processed} rows of Bitcoin data.')
            except Exception as e:
                messages.error(request, f'Error processing file: {str(e)}')
            
            return redirect('index')
    
    return redirect('index')

def historical(request):
    """Display historical data and charts"""
    df = get_historical_data()
    if df is None:
        messages.error(request, 'No Bitcoin data available. Please upload CSV data first.')
        return redirect('index')
    
    # Generate price chart
    price_chart = get_price_chart(df)
    
    # Get basic statistics
    stats = {
        'total_records': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'min_price': df['Close'].min(),
        'max_price': df['Close'].max(),
        'avg_price': df['Close'].mean(),
        'last_price': df['Close'].iloc[-1]
    }
    
    # Get recent predictions
    recent_predictions = PredictionResult.objects.all().order_by('-date')[:5]
    
    context = {
        'stats': stats,
        'price_chart': price_chart,
        'recent_predictions': recent_predictions
    }
    
    return render(request, 'predictor/historical.html', context)