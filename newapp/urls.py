from django.contrib import admin
from django.urls import path, include

from .views import upload_file 
from .views import draft
urlpatterns = [
    # path('check/',Predict),
    path('draft/',draft),
    path('upload/',upload_file)
    # path('admin/', admin.site.urls),
]