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

static bool save_pgm_frame(const uint8_t *data0);

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

    arv_camera_set_region(camera, 0, 0, width, height);
    //arv_camera_set_binning (camera, arv_option_horizontal_binning, arv_option_vertical_binning);
    //arv_camera_set_exposure_time (camera, arv_option_exposure_time_us);
    //arv_camera_set_gain (camera, arv_option_gain);
    
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
	int status;

	Py_BEGIN_ALLOW_THREADS;
	ArvBuffer *buffer;
	buffer = arv_stream_try_pop_buffer(stream);
	if (buffer != NULL) {
		if (buffer->status == ARV_BUFFER_STATUS_SUCCESS) {
                        unsigned BPP = ARV_PIXEL_FORMAT_BIT_PER_PIXEL(buffer->pixel_format);
                        save_pgm_frame(buffer->data);
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



static const char *save_basename(void)
{
    struct timeval tv;
    struct tm *tm;
    time_t t;
    unsigned hundredths;
    static char namebuf[100];

    gettimeofday(&tv, NULL);
    t = tv.tv_sec;
    tm = gmtime(&t);
    hundredths = tv.tv_usec / 10000;

    strftime(namebuf, sizeof(namebuf), "raw%Y%m%d%H%M%S", tm);
    snprintf(namebuf+strlen(namebuf), sizeof(namebuf)-strlen(namebuf), "%02uZ", hundredths);
    return namebuf;
}

/*
  save a .pgm greyscale image, auto-incrementing frame number
 */
static bool save_pgm_frame(const uint8_t *data0)
{
	int fd, ret;
	char *filename = NULL;
	uint16_t *data;
        uint32_t size = width*height*2;

	if (asprintf(&filename, "%s.pgm", save_basename()) <= 0) {
		return false;
	}

	printf("saving %s shape=%ux%u\n", 
	       filename,
	       (unsigned)width,
	       (unsigned)height);
  
	fd = open(filename, O_WRONLY|O_CREAT|O_TRUNC, 0644);
	if (fd == -1) {
		perror(filename);
		free(filename);
		return false;
	}
	free(filename);
	dprintf(fd,"P5\n%u %u\n%u\n", 
		width, height, 65535);

	data = malloc(size);
	memcpy(data, data0, size);

	// fix byte order if needed fpr pgm file
	if (__BYTE_ORDER == __LITTLE_ENDIAN) {
	  // needs byte swap
	  swab(data, data, size);
	}

	ret = write(fd, data, size);
	free(data);
	if (ret != size) {
		close(fd);
		printf("write failed ret=%d\n", ret);
		return false;
	}
	close(fd);
	return true;
}

static PyMethodDef TamariskMethods[] = {
  {"open", tamarisk_open, METH_VARARGS, "Open a tamarisk camera. Returns handle"},
  {"close", tamarisk_close, METH_VARARGS, "Close device."},
  {"capture", tamarisk_capture, METH_VARARGS, "Capture an image"},
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

