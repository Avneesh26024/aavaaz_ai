"""
URL configuration for Aavaaz project.
"""
from django.contrib import admin
from django.urls import path

from backend.websocket.views import scribe_token, health

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/scribe-token/', scribe_token, name='scribe_token'),
    path('api/health/', health, name='health'),
]


