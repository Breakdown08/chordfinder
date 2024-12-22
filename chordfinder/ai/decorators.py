from django.shortcuts import redirect

def user_not_authenticated(function=None, redirect_url='homepage_url'):
    """
    Decorator for views that checks that the user is NOT logged in, redirecting
    to the homepage if necessary by default.
    """
    def decorator(view_func):
        def _wrapped_view(request, *args, **kwargs):
            if request.user.is_authenticated:
                print("!!!NOOOOOOOOOOOT")
                return redirect(redirect_url)
                
            return view_func(request, *args, **kwargs)
            print("YESSSSSSSSSSSSSS")
        return _wrapped_view

    if function:
        print("FUNC@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", function)
        return decorator(function)
    print("ELSE@233223232323232332")

    return decorator