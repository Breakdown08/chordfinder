from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth import get_user_model


class UserRegistrationForm(UserCreationForm):
    errors_to_render = []
    username = forms.CharField( 
        error_messages = { 
        'required': "Пожалуйста введите имя пользователя",
        }) 
    email = forms.EmailField( 
        error_messages = { 
        'required': "Пожалуйста введите электронную почту пользователя",
        'invalid': "Некорректный Email адрес",
        })
    password1 = forms.CharField(
        widget=forms.PasswordInput(),
        error_messages = { 
            'required': "Пожалуйста придумайте ваш пароль",
            })
    password2 = forms.CharField(
        widget=forms.PasswordInput(),
        error_messages = { 
            'required': "Пожалуйста подтвердите ваш пароль",
            })


    def save(self, commit=True):
        user = super(UserRegistrationForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        if commit:
            user.save()

        return user


class UserLoginForm(AuthenticationForm):
    errors_to_render = []
    def __init__(self, *args, **kwargs):
        super(UserLoginForm, self).__init__(*args, **kwargs)

    username = forms.CharField(widget=forms.TextInput(
        attrs={'class': 'form-control', 'placeholder': 'Username or Email'}),
        label="Username or Email*")

    password = forms.CharField(widget=forms.PasswordInput(
        attrs={'class': 'form-control', 'placeholder': 'Password'}))


class PasswordResetForm(PasswordResetForm):
    def __init__(self, *args, **kwargs):
        super(PasswordResetForm, self).__init__(*args, **kwargs)
    email = forms.EmailField( 
        error_messages = { 
        'required': "Пожалуйста введите электронную почту пользователя",
        'invalid': "Некорректный Email адрес",
        })
    errors_to_render = []


class SetPasswordForm(SetPasswordForm):
    errors_to_render = []

    new_password1 = forms.CharField(
        widget=forms.PasswordInput(),
        error_messages = { 
            'required': "Пожалуйста введите новый пароль",
            })
    new_password2 = forms.CharField(
        widget=forms.PasswordInput(),
        error_messages = { 
            'required': "Пожалуйста подтвердите ваш новый пароль",
            })

    def __init__(self, *args, **kwargs):
        super(SetPasswordForm, self).__init__(*args, **kwargs)
