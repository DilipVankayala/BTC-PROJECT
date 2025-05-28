# predictor/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
    path('historical/', views.historical, name='historical'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
]
