/*
 * Face-value arithmetic for the fixed-width AD representation.
 *
 * The kernels are MRST's, from
 *   mrst-2026a/autodiff/ad-core/backends/diagonal/operators/mex/
 *     mexSinglePointUpwindDiagonalJac.cpp   (upwind gather)
 *     mexTwoPointGradientDiagonalJac.cpp    (two-point gradient)
 *     mexFaceAverageDiagonalJac.cpp         (face average)
 *     mexDiagMult.cpp                       (scale by a per-face value)
 *     mexDiagProductMult.cpp                (product rule)
 * (Copyright 2009-2026 SINTEF Digital, Mathematics & Cybernetics; MRST is
 * distributed under the GNU General Public License, which this file
 * inherits.)
 *
 * Changed from the originals: the MATLAB entry points are replaced by
 * Python ones, and only the row-major branch is kept -- that is the layout
 * PRSTCore's FaceValue uses, and carrying the column-major twin would mean
 * carrying two untested paths.
 *
 * Why compile these at all.  The divergence kernel next door made the
 * assembly competitive, and the remaining gap against the plain sparse
 * representation is here: a face value stores derivatives with respect to
 * both of its cells, six numbers per face for a three-variable black-oil
 * model, while the sparse form stores only the three that an upstream
 * gather actually depends on.  numpy therefore multiplies twenty percent
 * more numbers, and loses.  Compiled, the same arithmetic is a tight loop
 * over contiguous memory.
 *
 * Layout, throughout:
 *   cell data   (ncell, m)      C order, so entry (c, der) is [m*c + der]
 *   face data   (nface, 2, m)   C order, so (f, side, der) is [f*2*m + side*m + der]
 *   neighbours  (nface, 2)      C order, int64, zero-based
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>

typedef std::int64_t index_t;

/* MRST's upwindJac, row-major branch.  The upstream cell's derivatives go
 * into that side's half of the face row; the other half is zeroed, which is
 * what keeps every face value the same shape. */
static void upwindJac(const index_t nf, const index_t m,
                      const npy_bool* flag, const index_t* neighbours,
                      const double* cell_deriv, double* result)
{
    for (index_t face = 0; face < nf; face++) {
        const index_t take_left = flag[face] ? 1 : 0;
        const index_t copy_offset = take_left ? 0 : m;
        const index_t zero_offset = take_left ? m : 0;
        const index_t cell = neighbours[face * 2 + (take_left ? 0 : 1)];
        double* row = result + face * 2 * m;
        const double* source = cell_deriv + m * cell;
        for (index_t der = 0; der < m; der++) {
            row[copy_offset + der] = source[der];
            row[zero_offset + der] = 0.0;
        }
    }
}

/* MRST's gradientJac: the face value is right minus left, so the left cell
 * enters negated. */
static void gradientJac(const index_t nf, const index_t m,
                        const index_t* neighbours, const double* cell_deriv,
                        double* result)
{
    for (index_t face = 0; face < nf; face++) {
        const index_t left = neighbours[face * 2];
        const index_t right = neighbours[face * 2 + 1];
        double* row = result + face * 2 * m;
        for (index_t der = 0; der < m; der++) {
            row[der] = -cell_deriv[m * left + der];
            row[m + der] = cell_deriv[m * right + der];
        }
    }
}

/* MRST's faceAverageJac. */
static void faceAverageJac(const index_t nf, const index_t m,
                           const index_t* neighbours, const double* cell_deriv,
                           double* result)
{
    for (index_t face = 0; face < nf; face++) {
        const index_t left = neighbours[face * 2];
        const index_t right = neighbours[face * 2 + 1];
        double* row = result + face * 2 * m;
        for (index_t der = 0; der < m; der++) {
            row[der] = 0.5 * cell_deriv[m * left + der];
            row[m + der] = 0.5 * cell_deriv[m * right + der];
        }
    }
}

/* MRST's plain side gather: one cell of the face, without the upwind test.
 * Not a separate MEX file there -- it is upwindJac with a constant flag --
 * but a dedicated loop avoids materialising the flag array. */
static void sideJac(const index_t nf, const index_t m, const int side,
                    const index_t* neighbours, const double* cell_deriv,
                    double* result)
{
    const index_t copy_offset = side ? m : 0;
    const index_t zero_offset = side ? 0 : m;
    for (index_t face = 0; face < nf; face++) {
        const index_t cell = neighbours[face * 2 + side];
        double* row = result + face * 2 * m;
        const double* source = cell_deriv + m * cell;
        for (index_t der = 0; der < m; der++) {
            row[copy_offset + der] = source[der];
            row[zero_offset + der] = 0.0;
        }
    }
}

/* MRST's diagMult: scale every derivative of a face by that face's value. */
static void diagMult(const index_t n, const index_t width,
                     const double* v, const double* D, double* out)
{
    for (index_t i = 0; i < n; i++) {
        const double vi = v[i];
        const index_t base = i * width;
        for (index_t j = 0; j < width; j++) {
            out[base + j] = D[base + j] * vi;
        }
    }
}

/* MRST's diagProductMult: the product rule, v1*D1 + v2*D2. */
static void diagProductMult(const index_t n, const index_t width,
                            const double* v1, const double* D1,
                            const double* v2, const double* D2, double* out)
{
    for (index_t i = 0; i < n; i++) {
        const double a = v1[i];
        const double b = v2[i];
        const index_t base = i * width;
        for (index_t j = 0; j < width; j++) {
            out[base + j] = a * D1[base + j] + b * D2[base + j];
        }
    }
}

/* ------------------------------------------------------------------ glue -- */

static PyArrayObject* as_double(PyObject* obj, const char* name)
{
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_DOUBLE,
                                                        NPY_ARRAY_IN_ARRAY);
    if (a == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be convertible to float64", name);
    }
    return a;
}

static PyArrayObject* as_int64(PyObject* obj, const char* name)
{
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(obj, NPY_INT64,
                                                        NPY_ARRAY_IN_ARRAY);
    if (a == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be convertible to int64", name);
    }
    return a;
}

static PyObject* new_face_array(index_t nf, index_t m, double** data)
{
    npy_intp dims[3] = { (npy_intp)nf, 2, (npy_intp)m };
    PyObject* out = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    if (out != NULL) {
        *data = (double*)PyArray_DATA((PyArrayObject*)out);
    }
    return out;
}

/* Every gather shares the same argument handling; ``kind`` picks the loop. */
static PyObject* gather_common(PyObject* args, int kind)
{
    PyObject *cell_obj, *neighbours_obj, *flag_obj = NULL;
    if (kind == 0) {                                  /* upwind, needs a flag */
        if (!PyArg_ParseTuple(args, "OOO", &cell_obj, &neighbours_obj, &flag_obj)) {
            return NULL;
        }
    } else if (kind == 3 || kind == 4) {              /* left / right */
        if (!PyArg_ParseTuple(args, "OO", &cell_obj, &neighbours_obj)) {
            return NULL;
        }
    } else {                                          /* gradient, average */
        if (!PyArg_ParseTuple(args, "OO", &cell_obj, &neighbours_obj)) {
            return NULL;
        }
    }

    PyArrayObject* cell = as_double(cell_obj, "cell_deriv");
    PyArrayObject* neighbours = as_int64(neighbours_obj, "neighbours");
    PyArrayObject* flag = NULL;
    if (cell == NULL || neighbours == NULL) {
        Py_XDECREF(cell); Py_XDECREF(neighbours);
        return NULL;
    }
    if (PyArray_NDIM(cell) != 2 || PyArray_NDIM(neighbours) != 2 ||
        PyArray_DIM(neighbours, 1) != 2) {
        PyErr_SetString(PyExc_ValueError,
                        "cell_deriv must be (ncell, m) and neighbours (nface, 2)");
        Py_DECREF(cell); Py_DECREF(neighbours);
        return NULL;
    }
    const index_t nf = (index_t)PyArray_DIM(neighbours, 0);
    const index_t m = (index_t)PyArray_DIM(cell, 1);

    if (kind == 0) {
        flag = (PyArrayObject*)PyArray_FROM_OTF(flag_obj, NPY_BOOL,
                                                NPY_ARRAY_IN_ARRAY);
        if (flag == NULL || PyArray_SIZE(flag) != (npy_intp)nf) {
            PyErr_SetString(PyExc_ValueError, "flag must be a boolean array of nface");
            Py_XDECREF(flag); Py_DECREF(cell); Py_DECREF(neighbours);
            return NULL;
        }
    }

    double* result = NULL;
    PyObject* out = new_face_array(nf, m, &result);
    if (out == NULL) {
        Py_XDECREF(flag); Py_DECREF(cell); Py_DECREF(neighbours);
        return NULL;
    }

    const double* cell_p = (const double*)PyArray_DATA(cell);
    const index_t* nb_p = (const index_t*)PyArray_DATA(neighbours);

    Py_BEGIN_ALLOW_THREADS
    switch (kind) {
    case 0: upwindJac(nf, m, (const npy_bool*)PyArray_DATA(flag), nb_p, cell_p, result); break;
    case 1: gradientJac(nf, m, nb_p, cell_p, result); break;
    case 2: faceAverageJac(nf, m, nb_p, cell_p, result); break;
    case 3: sideJac(nf, m, 0, nb_p, cell_p, result); break;
    default: sideJac(nf, m, 1, nb_p, cell_p, result); break;
    }
    Py_END_ALLOW_THREADS

    Py_XDECREF(flag); Py_DECREF(cell); Py_DECREF(neighbours);
    return out;
}

static PyObject* upwind_jac(PyObject* self, PyObject* args)
{ (void)self; return gather_common(args, 0); }

static PyObject* two_point_gradient_jac(PyObject* self, PyObject* args)
{ (void)self; return gather_common(args, 1); }

static PyObject* face_average_jac(PyObject* self, PyObject* args)
{ (void)self; return gather_common(args, 2); }

static PyObject* left_jac(PyObject* self, PyObject* args)
{ (void)self; return gather_common(args, 3); }

static PyObject* right_jac(PyObject* self, PyObject* args)
{ (void)self; return gather_common(args, 4); }

static PyObject* diag_mult(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *deriv_obj, *v_obj;
    if (!PyArg_ParseTuple(args, "OO", &deriv_obj, &v_obj)) {
        return NULL;
    }
    PyArrayObject* deriv = as_double(deriv_obj, "deriv");
    PyArrayObject* v = as_double(v_obj, "values");
    if (deriv == NULL || v == NULL) {
        Py_XDECREF(deriv); Py_XDECREF(v);
        return NULL;
    }
    const index_t n = (index_t)PyArray_DIM(deriv, 0);
    if (PyArray_SIZE(v) != (npy_intp)n) {
        PyErr_SetString(PyExc_ValueError, "values must have one entry per row of deriv");
        Py_DECREF(deriv); Py_DECREF(v);
        return NULL;
    }
    const index_t width = (index_t)(PyArray_SIZE(deriv) / (n ? n : 1));

    PyObject* out = PyArray_SimpleNew(PyArray_NDIM(deriv), PyArray_DIMS(deriv),
                                      NPY_DOUBLE);
    if (out == NULL) {
        Py_DECREF(deriv); Py_DECREF(v);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    diagMult(n, width, (const double*)PyArray_DATA(v),
             (const double*)PyArray_DATA(deriv),
             (double*)PyArray_DATA((PyArrayObject*)out));
    Py_END_ALLOW_THREADS
    Py_DECREF(deriv); Py_DECREF(v);
    return out;
}

static PyObject* diag_product_mult(PyObject* self, PyObject* args)
{
    (void)self;
    PyObject *v1_obj, *d1_obj, *v2_obj, *d2_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &v1_obj, &d1_obj, &v2_obj, &d2_obj)) {
        return NULL;
    }
    PyArrayObject* v1 = as_double(v1_obj, "v1");
    PyArrayObject* d1 = as_double(d1_obj, "D1");
    PyArrayObject* v2 = as_double(v2_obj, "v2");
    PyArrayObject* d2 = as_double(d2_obj, "D2");
    if (!v1 || !d1 || !v2 || !d2) {
        Py_XDECREF(v1); Py_XDECREF(d1); Py_XDECREF(v2); Py_XDECREF(d2);
        return NULL;
    }
    const index_t n = (index_t)PyArray_DIM(d1, 0);
    if (PyArray_SIZE(d1) != PyArray_SIZE(d2) ||
        PyArray_SIZE(v1) != (npy_intp)n || PyArray_SIZE(v2) != (npy_intp)n) {
        PyErr_SetString(PyExc_ValueError, "mismatched shapes in the product rule");
        Py_DECREF(v1); Py_DECREF(d1); Py_DECREF(v2); Py_DECREF(d2);
        return NULL;
    }
    const index_t width = (index_t)(PyArray_SIZE(d1) / (n ? n : 1));

    PyObject* out = PyArray_SimpleNew(PyArray_NDIM(d1), PyArray_DIMS(d1), NPY_DOUBLE);
    if (out == NULL) {
        Py_DECREF(v1); Py_DECREF(d1); Py_DECREF(v2); Py_DECREF(d2);
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    diagProductMult(n, width,
                    (const double*)PyArray_DATA(v1), (const double*)PyArray_DATA(d1),
                    (const double*)PyArray_DATA(v2), (const double*)PyArray_DATA(d2),
                    (double*)PyArray_DATA((PyArrayObject*)out));
    Py_END_ALLOW_THREADS
    Py_DECREF(v1); Py_DECREF(d1); Py_DECREF(v2); Py_DECREF(d2);
    return out;
}

static PyMethodDef Methods[] = {
    {"upwind_jac", upwind_jac, METH_VARARGS,
     "upwind_jac(cell_deriv, neighbours, flag) -> (nface, 2, m)"},
    {"two_point_gradient_jac", two_point_gradient_jac, METH_VARARGS,
     "two_point_gradient_jac(cell_deriv, neighbours) -> (nface, 2, m), right minus left"},
    {"face_average_jac", face_average_jac, METH_VARARGS,
     "face_average_jac(cell_deriv, neighbours) -> (nface, 2, m), half of each side"},
    {"left_jac", left_jac, METH_VARARGS,
     "left_jac(cell_deriv, neighbours) -> (nface, 2, m), the first neighbour"},
    {"right_jac", right_jac, METH_VARARGS,
     "right_jac(cell_deriv, neighbours) -> (nface, 2, m), the second neighbour"},
    {"diag_mult", diag_mult, METH_VARARGS,
     "diag_mult(deriv, values) -> deriv scaled row by row"},
    {"diag_product_mult", diag_product_mult, METH_VARARGS,
     "diag_product_mult(v1, D1, v2, D2) -> v1*D1 + v2*D2, row by row"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "face_operators_ext",
    "Compiled face-value arithmetic (from MRST's diagonal-backend MEX kernels).",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_face_operators_ext(void)
{
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&Module);
}
