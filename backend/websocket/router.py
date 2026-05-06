from django.urls import path

from backend.websocket.consumer import TherapySessionConsumer

websocket_urlpatterns = [
    path("ws/session/", TherapySessionConsumer.as_asgi()),
]
