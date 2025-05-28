# predictor/forms.py

from django import forms

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label='Upload Bitcoin CSV Data')

class PredictionForm(forms.Form):
    MODELS = [
        ('logistic', 'Logistic Regression'),
        ('svc', 'Support Vector Classifier'),
        ('xgboost', 'XGBoost Classifier'),
    ]
    
    model_choice = forms.ChoiceField(choices=MODELS, label='Select Model')
