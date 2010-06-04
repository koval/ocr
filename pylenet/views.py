import base64
import os
import pickle
from StringIO import StringIO
import Image

from django.http import HttpResponse, Http404
from django.shortcuts import render_to_response

import numpy
import smala.datasets
import smala.arch

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

SIZE = 28
IMG = None
NN = None
PICKLE_FILE = os.path.join(os.path.dirname(__file__), 'datadigit/nn.pickle')

class Index(BaseView):

    def get(self, request):
        return render_to_response('pylenet/index.html')

class InitNN(BaseView):

    def get(self, request):
        global NN
        force = 'force' in request.GET
        if not force and os.path.exists(PICKLE_FILE):
            NN = pickle.load(file(PICKLE_FILE, 'rb'))
            return HttpResponse('Neural network loaded from file!')

        NN = smala.arch.lenet5()
        # Get dataset and neural network from the smala library
        dirname = os.path.dirname(__file__)
        images = os.path.join(dirname, 'datadigit/train-images-idx3-ubyte')
        labels = os.path.join(dirname, 'datadigit/train-labels-idx1-ubyte')
        if not (os.path.exists(images) and os.path.exists(labels)):
            return HttpResponse("Training files are absent!")

        ds = smala.datasets.MNIST(images, labels)
        ni = int(request.GET.get('i', 10))
        nj = int(request.GET.get('j', 100))

        for i in range(ni):
            for j in range(nj):
                # Training step
                X, T = ds.sample(j)
                NN.forward(X)
                NN.backward(X, NN.Y - T)
                NN.update(0.001)

        return HttpResponse('Teached %d times on %d images' % (ni, nj))

class DumpNN(BaseView):
    def get(self, request):
        if NN is None:
            return HttpResponse("Neural network isn't initialized!")
        response = HttpResponse(pickle.dumps(NN), mimetype='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename=nn.pickle'
        return response

class Recognize(BaseView):

    def post(self, request):
        # get image data in PNG format from request post body
        data = request.POST['i']
        data = base64.decodestring(data.split(',', 1)[-1])

        # resize, convert and invert image
        img = Image.open(StringIO(data))
        img = img.resize((SIZE, SIZE))
        img = img.convert('L')
        img = Image.eval(img, lambda p: 255-p)

        # store image data in a global variable to see it in a browser on GET request
        global IMG
        out = StringIO()
        img.save(out, 'JPEG')
        IMG = out.getvalue()

        global NN
        if NN is None and os.path.exists(PICKLE_FILE):
            NN = pickle.load(file(PICKLE_FILE, 'rb'))

        if NN is not None:
            # get array of image pixels
            pixels = list(img.getdata())

            X = numpy.zeros([32,32,1], dtype='f4', order='F')
            X[2:30,2:30,0] = numpy.array(pixels).reshape(SIZE, SIZE)
            X /= 100.0

            NN.forward(X)
            num = numpy.argmax(NN.Y)
            return HttpResponse(str(num))
        else:
            return HttpResponse('?')

    def get(self, request):
        if IMG is not None:
            resp = HttpResponse(IMG, mimetype='image/jpeg')
        else:
            resp = HttpResponse('No image')
        return resp

index = Index()
recognize = Recognize()
init = InitNN()
dump = DumpNN()
