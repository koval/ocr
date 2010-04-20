from django.conf.urls.defaults import *

import os

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    (r'^static/(?P<path>.*)$', 'django.views.static.serve',
        {'document_root': os.path.join(os.path.dirname(__file__), 'static')}),
    (r'^$', 'digitocr.pylenet.views.index'),
    (r'^init$', 'digitocr.pylenet.views.init'),
    (r'^dump$', 'digitocr.pylenet.views.dump'),
    (r'^recognize$', 'digitocr.pylenet.views.recognize'),

    # Example:
    # (r'^digitocr/', include('digitocr.foo.urls')),

    # Uncomment the admin/doc line below and add 'django.contrib.admindocs' 
    # to INSTALLED_APPS to enable admin documentation:
    # (r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # (r'^admin/', include(admin.site.urls)),
)
