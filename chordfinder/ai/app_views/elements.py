from django.shortcuts import render


def elements(request):
    data = {}
    return render(request, 'elements.html', context=data)
