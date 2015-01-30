#include <Python.h>
#include <numpy/arrayobject.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <arv.h>
#include <sys/time.h>
#include <time.h>

#define NUM_CAMERA_HANDLES 1

#ifndef Py_RETURN_NONE
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#endif

#define CHECK_CONTIGUOUS(a) do { if (!PyArray_ISCONTIGUOUS(a)) { \
	PyErr_SetString(TamariskError, "array must be contiguous"); \
	return NULL; \
	}} while (0)

static PyObject *TamariskError;
static ArvCamera *camera;
static ArvStream *stream;
static const uint32_t width = 640;
static const uint32_t height = 480;

/*
  write one camera device register
 */
static void cam_write_register(ArvCamera *camera, uint32_t address, uint32_t value)
{
  arv_device_write_register(arv_camera_get_device(camera), address, value, NULL);
}

static PyObject *
tamarisk_open(PyObject *self, PyObject *args)
{
    // look for first camera
    camera = arv_camera_new(NULL);
    if (camera == NULL) {
        PyErr_SetString(TamariskError, "No camera found");
        return NULL;
    }

    cam_write_register(camera, 0x12500, 640);
    cam_write_register(camera, 0x12510, 480);
    cam_write_register(camera, 0x12540, 17825829);
    cam_write_register(camera, 0x12550, 0);
    cam_write_register(camera, 0x11030, 0);
    cam_write_register(camera, 0x13100, 0);

    arv_camera_set_region(camera, 0, 0, -1, -1);
    arv_camera_set_binning (camera, -1, -1);
    arv_camera_set_exposure_time (camera, -1);
    arv_camera_set_gain (camera, -1);
    
    printf("vendor name         = %s\n", arv_camera_get_vendor_name (camera));
    printf("model name          = %s\n", arv_camera_get_model_name (camera));
    printf("device id           = %s\n", arv_camera_get_device_id (camera));

    stream = arv_camera_create_stream(camera, NULL, NULL);
    if (stream == NULL) {
        PyErr_SetString(TamariskError, "Unable to create arv stream");
        return NULL;
    }
    if (ARV_IS_GV_STREAM (stream)) {
        gint payload;
        payload = arv_camera_get_payload(camera);
        for (uint16_t i = 0; i < 50; i++) {
            arv_stream_push_buffer(stream, arv_buffer_new(payload, NULL));
        }
    }

    arv_camera_set_acquisition_mode(camera, ARV_ACQUISITION_MODE_CONTINUOUS);
    arv_camera_set_frame_rate(camera, 2.0);
    arv_camera_start_acquisition(camera);

    return PyLong_FromLong(0);
}

static PyObject *
tamarisk_capture(PyObject *self, PyObject *args)
{
	int handle = -1;
	int timeout_ms = 0;
	PyArrayObject* array = NULL;
	if (!PyArg_ParseTuple(args, "iiO", &handle, &timeout_ms, &array))
		return NULL;

	CHECK_CONTIGUOUS(array);

	if (handle != 0) {
            PyErr_SetString(TamariskError, "Invalid handle");
            return NULL;
	}

	int ndim = PyArray_NDIM(array);
	if (ndim != 2){
            PyErr_SetString(TamariskError, "Array has invalid number of dimensions");
            return NULL;
	}

	int w = PyArray_DIM(array, 1);
	int h = PyArray_DIM(array, 0);
	int stride = PyArray_STRIDE(array, 0);
	if (w != width || h != height || stride != width*2) {
            PyErr_SetString(TamariskError, "Invalid array dimensions should be 640x480x2");
            return NULL;
	}

	void* buf = PyArray_DATA(array);
	int status = -1;

	Py_BEGIN_ALLOW_THREADS;
	ArvBuffer *buffer;
	buffer = arv_stream_timeout_pop_buffer(stream, timeout_ms*1000);
	if (buffer != NULL) {
		status = buffer->status;
		if (buffer->status == ARV_BUFFER_STATUS_SUCCESS && 
                    buffer->data != NULL &&
                    ARV_PIXEL_FORMAT_BIT_PER_PIXEL(buffer->pixel_format) == 16) {
                    memcpy(buf, buffer->data, width*height*2);
                }
		arv_stream_push_buffer (stream, buffer);
	}
	Py_END_ALLOW_THREADS;
	
	if (status < 0) {
            PyErr_SetString(TamariskError, "Failed to capture");
            return NULL;
	}
	return Py_BuildValue("f", 0);
}


static PyObject *
tamarisk_close(PyObject *self, PyObject *args)
{
	int handle = -1;
	if (!PyArg_ParseTuple(args, "i", &handle))
		return NULL;

	if (handle != 0) {
            PyErr_SetString(TamariskError, "Invalid handle");
            return NULL;
	}
        arv_camera_stop_acquisition(camera);
        camera = NULL;
	Py_RETURN_NONE;
}


/* low level save routine */
static int _save_pgm(const char *filename, unsigned w, unsigned h, unsigned stride,
		     const char *data)
{
	int fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0644);
	int ret;

	if (fd == -1) {
		return -1;
	}
	dprintf(fd,"P5\n%u %u\n%u\n", 
		w, h, stride==w?255:65535);
	if (__BYTE_ORDER == __LITTLE_ENDIAN && stride == w*2) {
		char *data2 = malloc(w*h*2);
		swab(data, data2, w*h*2);
		ret = write(fd, data2, h*stride);
		free(data2);
	} else {
		ret = write(fd, data, h*stride);
	}
	if (ret != h*stride) {
		close(fd);
		return -1;
	}
	close(fd);
	return 0;
}

/*
  save a pgm image 
 */
static PyObject *
save_pgm(PyObject *self, PyObject *args)
{
	int status;
	const char *filename;
	unsigned w, h, stride;
	PyArrayObject* array = NULL;

	if (!PyArg_ParseTuple(args, "sO", &filename, &array))
		return NULL;

	CHECK_CONTIGUOUS(array);

	w = PyArray_DIM(array, 1);
	h = PyArray_DIM(array, 0);
	stride = PyArray_STRIDE(array, 0);

	Py_BEGIN_ALLOW_THREADS;
	status = _save_pgm(filename, w, h, stride, PyArray_DATA(array));
	Py_END_ALLOW_THREADS;
	if (status != 0) {
		PyErr_SetString(TamariskError, "pgm save failed");
		return NULL;
	}
	Py_RETURN_NONE;
}

static PyMethodDef TamariskMethods[] = {
  {"open", tamarisk_open, METH_VARARGS, "Open a tamarisk camera. Returns handle"},
  {"close", tamarisk_close, METH_VARARGS, "Close device."},
  {"capture", tamarisk_capture, METH_VARARGS, "Capture an image"},
  {"save_pgm", save_pgm, METH_VARARGS, "Save to a pgm gile"},
  {NULL, NULL, 0, NULL}        /* Terminus */
};

PyMODINIT_FUNC
inittamarisk(void)
{
  PyObject *m;

  m = Py_InitModule("tamarisk", TamariskMethods);
  if (m == NULL)
    return;

  import_array();

  TamariskError = PyErr_NewException("tamarisk.error", NULL, NULL);
  Py_INCREF(TamariskError);
  PyModule_AddObject(m, "error", TamariskError);
}

