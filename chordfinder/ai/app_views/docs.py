from django.shortcuts import render


def docs(request):
    data = {}
    return render(request, 'docs.html', context=data)
