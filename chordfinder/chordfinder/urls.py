from django.contrib import admin
from django.urls import path
from ai.views import predict_view, home

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('api/predict', predict_view, name='predict'),
]
