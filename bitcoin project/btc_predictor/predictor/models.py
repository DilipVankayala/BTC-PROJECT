from django.db import models

# Create your models here.

# predictor/models.py

from django.db import models

class BitcoinData(models.Model):
    date = models.DateField()
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    volume = models.FloatField()
    
    def __str__(self):
        return f"Bitcoin data for {self.date}"

class PredictionResult(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    prediction = models.BooleanField()  # True for price increase, False for decrease
    confidence = models.FloatField()
    model_used = models.CharField(max_length=100)
    
    def __str__(self):
        prediction_text = "increase" if self.prediction else "decrease"
        return f"Prediction: {prediction_text} with {self.confidence:.2f}% confidence on {self.date}"
