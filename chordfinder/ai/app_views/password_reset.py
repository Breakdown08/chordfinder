from django.shortcuts import render, redirect
from ..forms import PasswordResetForm, SetPasswordForm
from django.contrib.auth import get_user_model
from django.utils.encoding import force_bytes, force_str
from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.core.mail import EmailMessage
from django.db.models import Q
from ..tokens import account_activation_token
import json
from ..models import Profile
from django.conf import settings


def password_reset_request(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        form.errors_to_render = []
        if form.is_valid():
            user_email = form.cleaned_data['email']
            associated_user = get_user_model().objects.filter(Q(email=user_email)).first()
            if associated_user:
                if send_reset_password_request(request, associated_user):
                    return redirect('password_reset_request_success_url')
            else:
                form.errors_to_render.append("Пользователь не существует")
        errors = json.loads(form.errors.as_json())
        for field_error in errors.keys():
            for type_error in errors[field_error]:
                form.errors_to_render.append(type_error['message'])
    else:
        form = PasswordResetForm()
    return render(request, "password_reset_request.html", context={"form": form})


def send_reset_password_request(request, user):
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    token = account_activation_token.make_token(user)
    domain = get_current_site(request).domain if settings.ENV_NAME == 'local' else settings.SITE_URL
    protocol = 'http' if settings.ENV_NAME == 'local' else 'https'
    subject = "Запрос на смену пароля"
    url = "reset"
    message = render_to_string("template_reset_password.html", {
        'user': user,
        'domain': domain,
        'uid': uid,
        'token': token,
        "protocol": protocol
    })
    user_profile = Profile.objects.get(user=user)
    user_profile.password_reset_token = token
    user_profile.save()
    if settings.ENV_NAME == "production":
        email = EmailMessage(subject, message, to=[user.email])
        if email.send():
            return True
    else:
        print(f"{protocol}://{domain}/{url}/{uid}/{token}")
        return True
    return True
    

def password_reset_request_sent(request):
    data = {}
    return render(request, 'password_reset_request_success.html', context=data)


def password_reset_confirm(request, uidb64, token):
    ERROR_CODES = {
        "password_too_similar": "Пароль не должен совпадать с именем пользователя",
        "password_too_short": "Пароль слишком короткий (Минимум 8 символов)",
        "password_mismatch": "Пароли не совпадают",
        "password_too_common": "Пароль слишком простой",
        "unique": "Пользователь с таким именем уже существует",
    }
    User = get_user_model()
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except:
        user = None
    form = SetPasswordForm(user)
    user_profile = Profile.objects.get(user=user)
    if user is not None and account_activation_token.check_token(user, token) and account_activation_token.check_token(user, user_profile.password_reset_token):
        if request.method == 'POST':
            form = SetPasswordForm(user, request.POST)
           
            if form.is_valid():
                form.save()
                user_profile.password_reset_token = ""
                user_profile.save()
                return redirect('homepage_url')
            else:
                form.errors_to_render = []
                errors = json.loads(form.errors.as_json())
                for field_error in errors.keys():
                    print(field_error, errors[field_error])
                    for type_error in errors[field_error]:
                        code = type_error["code"]
                        if str(code).__contains__("password"):
                            print(type_error)
                            form.errors_to_render.append(ERROR_CODES[code])
                        elif str(code).__contains__("unique"):
                            form.errors_to_render.append(ERROR_CODES[code])
                        else:
                            form.errors_to_render.append(type_error['message'])
        return render(request, 'password_reset_confirm.html', {'form': form})
    else:
        form.errors_to_render.append("Ссылка утратила срок действия")
    return redirect("homepage_url")