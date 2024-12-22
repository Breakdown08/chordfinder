from django.shortcuts import render


def homepage(request):
    data = {}
    return render(request, 'homepage.html', context=data)