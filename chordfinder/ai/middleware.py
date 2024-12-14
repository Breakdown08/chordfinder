from django.http import JsonResponse
from ai.models import APIKey


class APIKeyMiddleware:


    def __init__(self, get_response):
        self.get_response = get_response


    def __call__(self, request):
        # Проверяем только для API путей
        if request.path.startswith('/api/'):
            # Проверяем, есть ли заголовки
            api_key = request.headers.get('Authorization', '').split('Bearer ')[-1] if hasattr(request, 'headers') else None
            if not api_key or not APIKey.objects.filter(key=api_key, is_active=True).exists():
                return JsonResponse({'error': 'Invalid or missing API key'}, status=403)

        # Продолжаем обработку запроса
        return self.get_response(request)
