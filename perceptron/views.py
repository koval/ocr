import base64
import os
import pickle
from StringIO import StringIO
import Image

from django.http import HttpResponse, Http404
from django.shortcuts import render_to_response

from teacher import Teacher, get_in_vector
from perceptron import Perceptron

class BaseView(object):
    """ A base class for views that delegates handling to GET, POST methods
        based on reques method. An instance of this class must be created to be
        used in the url patterns.
    """

    def __call__(self, request, *args, **kwargs):
        method = request.method.lower()
        if hasattr(self, method):
            return getattr(self, method)(request, *args, **kwargs)
        else:
            raise Http404('Not supported HTTP method.')

    def error(self, status):
        raise Http404('Not found')

class Index(BaseView):

    def get(self, request):
        return render_to_response('perceptron/index.html')

IMG = None
SIZE = 64
PICKLE_FILE = os.path.join(os.path.dirname(__file__), 'knowledge%d' % SIZE)

class Recognize(BaseView):

    def post(self, request):
        # get image data in PNG format from request post body
        data = request.POST['i']
        data = base64.decodestring(data.split(',', 1)[-1])

        # resize, convert and invert image
        img = Image.open(StringIO(data))
        img = img.resize((SIZE, SIZE))

        # store image data in a global variable to see it in a browser on GET request
        global IMG
        out = StringIO()
        img.save(out, 'JPEG')
        IMG = out.getvalue()

        pixels = list(img.getdata())
        perceptron = pickle.load(file(PICKLE_FILE, 'rb'))
        vector = get_in_vector(pixels)
        out = perceptron.recognize(vector, raw=True)
        m = max(out)
        num = []
        n = 0
        for v in out:
            if v == m:
                num.append(str(n))
            n += 1
        num = ','.join(num)

        return HttpResponse(str(num))

    def get(self, request):
        if IMG is not None:
            resp = HttpResponse(IMG, mimetype='image/jpeg')
        else:
            resp = HttpResponse('No image')
        return resp

index = Index()
recognize = Recognize()
