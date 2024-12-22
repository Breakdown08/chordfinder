from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from ..forms import UserLoginForm

from django.contrib import messages
import json


def sign_in(request):
    if request.method == "POST":
        form = UserLoginForm(request=request, data=request.POST)
        if form.is_valid():
            user = authenticate(
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password"],
            )
            if user is not None:
                login(request, user)
                messages.success(request, f"Hello <b>{user.username}</b>! You have been logged in")
                return redirect("homepage_url")

        else:
            errors = json.loads(form.errors.as_json())
            form.errors_to_render = []
            for field_error in errors.keys():
                if errors[field_error][0]["code"] == "invalid_login":
                    form.errors_to_render.append("Неверный логин или пароль")
    else:
        form = UserLoginForm()

    return render(request, "sign_in.html", context={"form": form})


