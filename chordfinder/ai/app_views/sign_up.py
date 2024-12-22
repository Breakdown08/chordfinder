from ..forms import UserRegistrationForm
from django.contrib.sites.shortcuts import get_current_site
from ..tokens import account_activation_token
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes
from django.contrib.auth import get_user_model
from django.utils.http import urlsafe_base64_encode
from django.db.models import Q
from django.shortcuts import render, redirect
from django.core.mail import EmailMessage
from django.conf import settings
import json


def sign_up(request):
    ERROR_CODES = {
        "password_too_similar": "Пароль не должен совпадать с именем пользователя",
        "password_too_short": "Пароль слишком короткий (Минимум 8 символов)",
        "password_mismatch": "Пароли не совпадают",
        "password_too_common": "Пароль слишком простой",
        "unique": "Пользователь с таким именем уже существует",
    }

    if request.method == "POST":
        form = UserRegistrationForm(request.POST)
        form.errors_to_render = []
        if form.is_valid():
            associated_user = get_user_model().objects.filter(Q(email=form.cleaned_data.get('email'))).first()
            if associated_user:
                form.errors_to_render = ["Пользователь с таким Email уже зарегистрирован"]
            else:
                user = form.save(commit=False)
                user.is_active=False
                user.save()
                if send_activation_link(request, user, form.cleaned_data.get('email')):
                    return redirect('/sign_up/success')
        else:
            errors = json.loads(form.errors.as_json())
            print("СПИСОК ОШИБОК:")
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
    else:
        form = UserRegistrationForm()

    return render(request, "sign_up.html", context={"form": form})


def send_activation_link(request, user, to_email):
    mail_subject = f"Активация аккаунта {user.username}"
    token = account_activation_token.make_token(user)
    uid = urlsafe_base64_encode(force_bytes(user.pk))
    domain = get_current_site(request).domain
    protocol = 'https' if request.is_secure() else 'http'
    url = "activate_account"
    message = render_to_string("template_activate_account.html", {
        'user': user.username,
        'domain': domain,
        'url': url,
        'uid': uid,
        'token': token,
        "protocol": protocol
    })
    

    if settings.ENV_NAME == "production":
        email = EmailMessage(mail_subject, message, to=[to_email], from_email=settings.EMAIL_FROM)
        if email.send():
            return True
        else:
            return False
    else:
        print("Токен отправлен", f"{protocol}://{domain}/{url}/{uid}/{token}")
        return True


def sign_up_success(request):
    data = {}
    return render(request, 'sign_up_success.html', context=data)
