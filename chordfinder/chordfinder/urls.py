from django.contrib import admin
from django.urls import path, include
from django.urls import path
from ai.app_views import activate_account, elements, homepage, profile, sign_up, sign_in, password_reset, predict, docs

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', homepage.homepage, name='homepage_url'),
    path('docs', docs.docs, name='docs_url'),
    path('api/predict', predict.predict, name='predict_url'),
    path('accounts/', include("django.contrib.auth.urls")),
    path('accounts/profile/', profile.profile, name='profile_url'),
    path('elements', elements.elements, name='elements_url'),
    path('sign_in', sign_in.sign_in, name='sign_in_url'),
    path('sign_up/', sign_up.sign_up, name='sign_up_url'),
    path('sign_up/success', sign_up.sign_up_success, name="sign_up_success_url"),
    path('activate_account/<uidb64>/<token>', activate_account.activate_account, name='activate_account_url'),
    path("password_change", profile.password_change, name="password_change_url"),
    path("password_reset", password_reset.password_reset_request, name="password_reset_request_url"),
    path("password_reset/sent", password_reset.password_reset_request_sent, name="password_reset_request_success_url"),
    path('reset/<uidb64>/<token>', password_reset.password_reset_confirm, name='password_reset_confirm_url'),
]
