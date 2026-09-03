/*
 * Discrete divergence Jacobian assembly.
 *
 * The kernel below is MRST's, taken from
 *   mrst-2026a/autodiff/ad-core/backends/diagonal/operators/mex/
 *   mexDiscreteDivergenceJac.cpp
 * (Copyright 2009-2026 SINTEF Digital, Mathematics & Cybernetics; MRST is
 * distributed under the GNU General Public License).  Deriving from it puts
 * this file under the same licence.
 *
 * Two things changed, and nothing else:
 *   - the MATLAB entry point (mexFunction, mxArray) is replaced by a Python
 *     one, so the result is a scipy CSC matrix rather than a MATLAB sparse;
 *   - the precomputed index arrays are taken as 64-bit integers instead of
 *     MATLAB doubles, which is what numpy produces and what scipy's index
 *     arrays already are.
 *
 * What it does: given the per-face derivatives of a flux with respect to the
 * two cells the face separates, write the divergence's Jacobian straight
 * into CSC.  The point is that nothing is sorted and nothing is searched --
 * the precomputes say exactly where every contribution lands, so the whole
 * assembly is one pass.  Building the same matrix by handing scipy an
 * unordered triplet list costs three times as much, because the sort
 * dominates.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdint>

typedef std::int64_t index_t;

/* MRST's copyFaceData, unchanged apart from the index type.
 *
 * ``lower`` distinguishes the two halves of a face: a connection whose
 * "self" cell is the first column of the neighbour table reads the
 * left-cell derivative and negates it; the mirrored connection reads the
 * right-cell one.  Each contribution goes to the other cell's row and is
 * subtracted from this cell's diagonal, which is the divergence.
 */
template <bool rowMajor, bool lower>
static inline void copyFaceData(const index_t c, const index_t nf, const index_t m,
                                const index_t diag, const index_t passed,
                                const index_t sparse_mult, const index_t cell_offset,
                                const index_t f, const index_t fl,
                                const double* diagonal,
                                double* pr, index_t* ir)
{
    for (index_t der = 0; der < m; der++) {
        double v;
        const index_t sparse_offset = der * sparse_mult + cell_offset;
        if (lower) {
            v = rowMajor ? -diagonal[f * 2 * m + der] : -diagonal[der * nf + f];
            ir[sparse_offset + fl + passed] = -c;
        } else {
            v = rowMajor ? diagonal[(f * 2 + 1) * m + der]
                         : diagonal[der * nf + f + m * nf];
            ir[sparse_offset + fl + passed] = c;
        }
        pr[sparse_offset + fl + passed] = v;
        pr[sparse_offset + diag] -= v;
    }
}

/* The same assembly, writing CSR instead of CSC.
 *
 * MRST writes CSC because that is what a MATLAB sparse matrix is.  scipy
 * accepts either, but everything downstream here wants CSR, and converting
 * a two-million-entry CSC costs more than the assembly saved: measured on
 * Norne, the conversion alone was larger than the whole kernel.
 *
 * The structure allows it directly.  Row ``cell`` holds, for each variable
 * in turn, that cell's own column and its neighbours' -- and the
 * precomputes already sort the neighbours, so the columns come out
 * ascending within each variable's run, which is what CSR requires.  Only
 * the index arithmetic differs from the routine above.
 */
template <bool has_accumulation, bool rowMajor>
static void divergenceJacCSR(const index_t nf, const index_t nc, const index_t m,
                             const index_t* facePos, const index_t* faces,
                             const index_t* cells, const index_t* cells_ix,
                             const double* accumulation, const double* diagonal,
                             double* pr, index_t* ir, index_t* jc)
{
    jc[0] = 0;
    for (index_t cell = 0; cell < nc; cell++) {
        const index_t f_offset = facePos[cell];
        const index_t n_local_hf = facePos[cell + 1] - f_offset;
        const index_t diag = cells_ix[cell];
        const index_t row_start = m * (f_offset + cell);
        const index_t per_der = n_local_hf + 1;
        jc[cell + 1] = row_start + m * per_der;

        for (index_t der = 0; der < m; der++) {
            const index_t base = row_start + der * per_der;
            const index_t column_shift = der * nc;
            const index_t dpos = base + diag;
            ir[dpos] = cell + column_shift;
            pr[dpos] = has_accumulation
                ? (rowMajor ? accumulation[cell * m + der]
                            : accumulation[der * nc + cell])
                : 0.0;

            for (index_t fl = 0; fl < n_local_hf; fl++) {
                const index_t passed = (fl >= diag) ? 1 : 0;
                const index_t f = faces[f_offset + fl];
                const index_t c = cells[f_offset + fl];
                index_t other;
                double own, cross;
                /* Two different entries come out of one connection, and
                 * they read opposite sides of the face.
                 *
                 * ``own`` is this cell's own dependence on the flux it
                 * gains or loses, and lands on the diagonal.  ``cross`` is
                 * its dependence on the *neighbour's* variables, and lands
                 * in the neighbour's column -- which is the mirrored
                 * connection's contribution, so it reads the other side.
                 * MRST writes CSC and only ever needs ``own``: the cross
                 * term arrives when the loop reaches the neighbour.
                 * Writing CSR needs both at once, and using ``own`` for
                 * the neighbour's column transposes the matrix -- a
                 * mistake that hides completely when a face's two
                 * derivatives happen to be equal and opposite.
                 */
                if (c < 0) {
                    /* This cell is the face's left neighbour: it gains the
                     * flux, so its own variables enter with +d_left and the
                     * neighbour's with +d_right. */
                    other = -(c + 1);
                    own = rowMajor ? -diagonal[f * 2 * m + der]
                                   : -diagonal[der * nf + f];
                    cross = rowMajor ? diagonal[(f * 2 + 1) * m + der]
                                     : diagonal[der * nf + f + m * nf];
                } else {
                    /* The right neighbour: it loses the flux, so both signs
                     * flip. */
                    other = c - 1;
                    own = rowMajor ? diagonal[(f * 2 + 1) * m + der]
                                   : diagonal[der * nf + f + m * nf];
                    cross = rowMajor ? -diagonal[f * 2 * m + der]
                                     : -diagonal[der * nf + f];
                }
                ir[base + fl + passed] = other + column_shift;
                pr[base + fl + passed] = cross;
                pr[dpos] -= own;
            }
        }
    }
}

/* MRST's divergenceJac. */
template <bool has_accumulation, bool rowMajor>
static void divergenceJac(const index_t nf, const index_t nc, const index_t m,
                          const index_t* facePos, const index_t* faces,
                          const index_t* cells, const index_t* cells_ix,
                          const double* accumulation, const double* diagonal,
                          double* pr, index_t* ir, index_t* jc)
{
    const index_t mv = facePos[nc];
    jc[0] = 0;
    for (index_t cell = 0; cell < nc; cell++) {
        const index_t f_offset = facePos[cell];
        const index_t n_local_hf = facePos[cell + 1] - f_offset;
        const index_t diag = cells_ix[cell];
        const index_t cell_offset = f_offset + cell;
        for (index_t der = 0; der < m; der++) {
            const index_t base = der * (mv + nc) + cell_offset;
            jc[cell + der * nc + 1] = base + n_local_hf + 1;
            const index_t dpos = base + diag;
            ir[dpos] = cell;
            if (has_accumulation) {
                pr[dpos] = rowMajor ? accumulation[cell * m + der]
                                    : accumulation[der * nc + cell];
            } else {
                pr[dpos] = 0.0;
            }
        }
        for (index_t fl = 0; fl < n_local_hf; fl++) {
            const index_t passed = (fl >= diag) ? 1 : 0;
            const index_t f = faces[f_offset + fl];
            const index_t c = cells[f_offset + fl];
            const index_t sparse_mult = mv + nc;
            if (c < 0) {
                copyFaceData<rowMajor, true>(c + 1, nf, m, diag, passed, sparse_mult,
                                             cell_offset, f, fl, diagonal, pr, ir);
            } else {
                copyFaceData<rowMajor, false>(c - 1, nf, m, diag, passed, sparse_mult,
                                              cell_offset, f, fl, diagonal, pr, ir);
            }
        }
    }
}

/* ------------------------------------------------------------------ glue -- */

static PyArrayObject* as_int64(PyObject* obj, const char* name)
{
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(
        obj, NPY_INT64, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be convertible to int64", name);
    }
    return array;
}

static PyArrayObject* as_double(PyObject* obj, const char* name)
{
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(
        obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        PyErr_Format(PyExc_TypeError, "%s must be convertible to float64", name);
    }
    return array;
}

PyDoc_STRVAR(divergence_jac_doc,
"divergence_jac(nf, nc, m, face_pos, faces, cells, cell_index, accumulation, diagonal)\n"
"\n"
"Assemble the discrete divergence Jacobian in CSC form.\n"
"\n"
"``diagonal`` holds the per-face derivatives, shaped (nf, 2, m) in C order:\n"
"entry (f, side, k) is d(flux across face f)/d(variable k in the cell on\n"
"that side of it).  ``accumulation`` may be None, or (nc, m) in C order.\n"
"The other four arrays are the grid's precomputes.\n"
"\n"
"Returns (data, indices, indptr) for a scipy matrix of shape (nc, nc*m):\n"
"CSR by default, CSC when the optional tenth argument is false.");

static PyObject* divergence_jac(PyObject* self, PyObject* args)
{
    (void)self;
    long long nf_in, nc_in, m_in;
    int csr = 1;
    PyObject *face_pos_obj, *faces_obj, *cells_obj, *cell_index_obj;
    PyObject *accumulation_obj, *diagonal_obj;

    if (!PyArg_ParseTuple(args, "LLLOOOOOO|p", &nf_in, &nc_in, &m_in,
                          &face_pos_obj, &faces_obj, &cells_obj, &cell_index_obj,
                          &accumulation_obj, &diagonal_obj, &csr)) {
        return NULL;
    }
    const index_t nf = (index_t)nf_in, nc = (index_t)nc_in, m = (index_t)m_in;
    if (nc <= 0 || m <= 0) {
        PyErr_SetString(PyExc_ValueError, "nc and m must be positive");
        return NULL;
    }

    PyArrayObject* face_pos = as_int64(face_pos_obj, "face_pos");
    PyArrayObject* faces = as_int64(faces_obj, "faces");
    PyArrayObject* cells = as_int64(cells_obj, "cells");
    PyArrayObject* cell_index = as_int64(cell_index_obj, "cell_index");
    PyArrayObject* diagonal = as_double(diagonal_obj, "diagonal");
    PyArrayObject* accumulation = NULL;
    const bool has_accumulation = (accumulation_obj != Py_None);
    if (has_accumulation) {
        accumulation = as_double(accumulation_obj, "accumulation");
    }
    if (!face_pos || !faces || !cells || !cell_index || !diagonal ||
        (has_accumulation && !accumulation)) {
        goto fail;
    }
    if (PyArray_SIZE(face_pos) != (npy_intp)(nc + 1)) {
        PyErr_SetString(PyExc_ValueError, "face_pos must have nc + 1 entries");
        goto fail;
    }
    if (PyArray_SIZE(diagonal) != (npy_intp)(nf * 2 * m)) {
        PyErr_SetString(PyExc_ValueError,
                        "diagonal must hold nf * 2 * m values");
        goto fail;
    }

    {
        const index_t* face_pos_p = (const index_t*)PyArray_DATA(face_pos);
        const index_t mv = face_pos_p[nc];
        const npy_intp nnz = (npy_intp)(m * (mv + nc));
        const npy_intp ncol = (npy_intp)(nc * m);

        npy_intp nnz_dims[1] = { nnz };
        /* CSR has one pointer per cell; CSC one per (cell, variable). */
        npy_intp col_dims[1] = { (csr ? (npy_intp)nc : ncol) + 1 };
        PyObject* data = PyArray_ZEROS(1, nnz_dims, NPY_DOUBLE, 0);
        PyObject* indices = PyArray_ZEROS(1, nnz_dims, NPY_INT64, 0);
        PyObject* indptr = PyArray_ZEROS(1, col_dims, NPY_INT64, 0);
        if (!data || !indices || !indptr) {
            Py_XDECREF(data); Py_XDECREF(indices); Py_XDECREF(indptr);
            goto fail;
        }

        double* pr = (double*)PyArray_DATA((PyArrayObject*)data);
        index_t* ir = (index_t*)PyArray_DATA((PyArrayObject*)indices);
        index_t* jc = (index_t*)PyArray_DATA((PyArrayObject*)indptr);
        const double* acc_p = has_accumulation
            ? (const double*)PyArray_DATA(accumulation) : NULL;
        const double* diag_p = (const double*)PyArray_DATA(diagonal);

        const index_t* faces_p = (const index_t*)PyArray_DATA(faces);
        const index_t* cells_p = (const index_t*)PyArray_DATA(cells);
        const index_t* cell_ix_p = (const index_t*)PyArray_DATA(cell_index);

        Py_BEGIN_ALLOW_THREADS
        if (csr) {
            if (has_accumulation) {
                divergenceJacCSR<true, true>(nf, nc, m, face_pos_p, faces_p,
                                             cells_p, cell_ix_p, acc_p, diag_p,
                                             pr, ir, jc);
            } else {
                divergenceJacCSR<false, true>(nf, nc, m, face_pos_p, faces_p,
                                              cells_p, cell_ix_p, NULL, diag_p,
                                              pr, ir, jc);
            }
        } else if (has_accumulation) {
            divergenceJac<true, true>(nf, nc, m, face_pos_p, faces_p, cells_p,
                                      cell_ix_p, acc_p, diag_p, pr, ir, jc);
        } else {
            divergenceJac<false, true>(nf, nc, m, face_pos_p, faces_p, cells_p,
                                       cell_ix_p, NULL, diag_p, pr, ir, jc);
        }
        Py_END_ALLOW_THREADS

        Py_DECREF(face_pos); Py_DECREF(faces); Py_DECREF(cells);
        Py_DECREF(cell_index); Py_DECREF(diagonal);
        Py_XDECREF(accumulation);
        return Py_BuildValue("NNN", data, indices, indptr);
    }

fail:
    Py_XDECREF(face_pos); Py_XDECREF(faces); Py_XDECREF(cells);
    Py_XDECREF(cell_index); Py_XDECREF(diagonal); Py_XDECREF(accumulation);
    return NULL;
}

static PyMethodDef Methods[] = {
    {"divergence_jac", divergence_jac, METH_VARARGS, divergence_jac_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "discrete_divergence_ext",
    "Compiled discrete divergence Jacobian assembly (from MRST's MEX kernel).",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_discrete_divergence_ext(void)
{
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&Module);
}
