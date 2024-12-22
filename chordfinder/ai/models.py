from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import secrets


class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    password_reset_token = models.CharField(max_length=256, default="")
    api_key = models.CharField(max_length=64, unique=True, default=secrets.token_urlsafe)

    def __str__(self):
        return self.user.username


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
