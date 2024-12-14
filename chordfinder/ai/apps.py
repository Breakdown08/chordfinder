from django.apps import AppConfig


class AiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai'


    def ready(self):
        from ai.neural_network.model import start
        start()
