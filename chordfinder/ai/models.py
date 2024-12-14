from django.db import models
import secrets


class APIKey(models.Model):
    key = models.CharField(max_length=64, unique=True, default=secrets.token_urlsafe)
    owner = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
