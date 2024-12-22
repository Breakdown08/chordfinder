from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import EmailMessage
from django.contrib.auth import get_user_model
from ..models import Profile
from ..forms import SetPasswordForm


def profile(request):
    if not request.user.is_authenticated:
        return redirect("sign_in_url")
    data = {"api_key": Profile.objects.filter(user=request.user).first().api_key}
    return render(request, 'profile.html', context=data)


def password_change(request):
    if not request.user.is_authenticated:
        return redirect("sign_in_url")
    user = request.user
    if request.method == 'POST':
        form = SetPasswordForm(user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Your password has been changed")
            return redirect('sign_in_url')
        else:
            for error in list(form.errors.values()):
                messages.error(request, error)

    form = SetPasswordForm(user)
    return render(request, 'password_reset_confirm.html', {'form': form})