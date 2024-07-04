
from django.contrib import admin
from django.urls import path, include
from . import views as base_views

from .import views


from accounts import urls as accounts_urls

from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', base_views.HomePage.as_view(), name='home'),
    path('accounts/', include(accounts_urls)),
    path('detection/', views.DETECTION_PAGE, name='detection'), 
    path('detection_SVM/', views.DETECTION_PAGE_SVM, name='detection_SVM'), 

    path('info/', views.info, name='info'), 
    path('about_us/', views.about_us, name='about_us'), 
    path('display_result', views.display_result, name = 'display_result'),
    path('display_result_SVM', views.display_result_SVM, name = 'display_result_SVM'),


]

if settings.DEBUG:
        urlpatterns += static(settings.STATIC_URL,document_root=settings.STATICFILES_DIRS)

