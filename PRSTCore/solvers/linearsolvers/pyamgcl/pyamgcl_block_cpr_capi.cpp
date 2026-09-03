#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <chrono>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/runtime.hpp>
#include <amgcl/relaxation/runtime.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/runtime.hpp>
#include <amgcl/preconditioner/cpr.hpp>
#include <amgcl/preconditioner/cpr_drs.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/adapter/zero_copy.hpp>

template <typename T>
static bool copy_buffer(PyObject *obj, std::vector<T> &out, const char *name) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_CONTIG_RO) != 0) {
        return false;
    }
    if (view.itemsize != static_cast<Py_ssize_t>(sizeof(T))) {
        PyErr_Format(PyExc_TypeError, "%s itemsize must be %d", name, static_cast<int>(sizeof(T)));
        PyBuffer_Release(&view);
        return false;
    }
    if (view.len % view.itemsize != 0) {
        PyErr_Format(PyExc_ValueError, "%s buffer length is not aligned", name);
        PyBuffer_Release(&view);
        return false;
    }
    const T *first = static_cast<const T *>(view.buf);
    out.assign(first, first + view.len / view.itemsize);
    PyBuffer_Release(&view);
    return true;
}

static boost::property_tree::ptree make_cpr_prm(
    double tolerance, int max_iter, int block_size, int active_rows,
    const std::string &solver, const std::string &coarsening,
    const std::string &relaxation, const std::string &s_relaxation,
    bool use_drs, double drs_eps_dd, double drs_eps_ps,
    double aggr_eps_strong, double aggr_over_interp, double aggr_relax,
    int npre, int npost, int ncycle, bool direct_coarse,
    int coarse_enough, int max_levels, double ilu_damping,
    int iluk_k, int ilut_p, double ilut_tau, int gmres_m
) {
    boost::property_tree::ptree prm;
    prm.put("solver.type", solver);
    prm.put("solver.tol", tolerance);
    prm.put("solver.check_after", true);
    if (max_iter > 0) prm.put("solver.maxiter", max_iter);
    if (gmres_m > 0 && (solver == "gmres" || solver == "fgmres" || solver == "lgmres")) {
        prm.put("solver.M", gmres_m);
    }

    prm.put("precond.block_size", block_size);
    prm.put("precond.active_rows", active_rows);
    prm.put("precond.pprecond.coarsening.type", coarsening);
    prm.put("precond.pprecond.coarsening.aggr.eps_strong", aggr_eps_strong);
    prm.put("precond.pprecond.coarsening.over_interp", aggr_over_interp);
    // Keep the runtime AMGCL parameter tree close to the set understood by
    // the vendored AMGCL version.  MRST exposes additional knobs, but passing
    // unsupported keys here only produces noisy warnings without affecting the
    // block CPR solve.
    prm.put("precond.pprecond.direct_coarse", direct_coarse);
    if (coarse_enough >= 0) prm.put("precond.pprecond.coarse_enough", coarse_enough);
    if (max_levels >= 0) prm.put("precond.pprecond.max_levels", max_levels);
    if (npre >= 0) prm.put("precond.pprecond.npre", npre);
    if (npost >= 0) prm.put("precond.pprecond.npost", npost);
    if (ncycle >= 0) prm.put("precond.pprecond.ncycle", ncycle);

    prm.put("precond.pprecond.relax.type", relaxation);
    if (relaxation == "ilu0" || relaxation == "iluk" || relaxation == "ilut") {
        prm.put("precond.pprecond.relax.damping", ilu_damping);
    }
    if (relaxation == "iluk") prm.put("precond.pprecond.relax.k", iluk_k);
    if (relaxation == "ilut") {
        prm.put("precond.pprecond.relax.p", ilut_p);
        prm.put("precond.pprecond.relax.tau", ilut_tau);
    }
    prm.put("precond.sprecond.type", s_relaxation);
    if (s_relaxation == "ilu0" || s_relaxation == "iluk" || s_relaxation == "ilut") {
        prm.put("precond.sprecond.damping", ilu_damping);
    }
    if (s_relaxation == "iluk") prm.put("precond.sprecond.k", iluk_k);
    if (s_relaxation == "ilut") {
        prm.put("precond.sprecond.p", ilut_p);
        prm.put("precond.sprecond.tau", ilut_tau);
    }
    if (use_drs) {
        prm.put("precond.eps_dd", drs_eps_dd);
        prm.put("precond.eps_ps", drs_eps_ps);
    }
    return prm;
}

template <int B>
static PyObject *solve_block_cpr_t(
    const std::vector<int> &ptr,
    const std::vector<int> &col,
    const std::vector<double> &val,
    const std::vector<double> &rhs,
    double tolerance, int max_iter, int active_rows,
    const std::string &solver, const std::string &coarsening,
    const std::string &relaxation, const std::string &s_relaxation,
    bool use_drs, double drs_eps_dd, double drs_eps_ps,
    double aggr_eps_strong, double aggr_over_interp, double aggr_relax,
    int npre, int npost, int ncycle, bool direct_coarse,
    int coarse_enough, int max_levels, double ilu_damping,
    int iluk_k, int ilut_p, double ilut_tau, int gmres_m
) {
    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;
    typedef amgcl::backend::builtin<val_type> BlockBackend;
    typedef amgcl::backend::builtin<double> ScalarBackend;
    typedef amgcl::amg<ScalarBackend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper> PPrecond;
    typedef amgcl::relaxation::as_preconditioner<BlockBackend, amgcl::runtime::relaxation::wrapper> SPrecond;
    typedef amgcl::make_solver<amgcl::preconditioner::cpr<PPrecond, SPrecond>, amgcl::runtime::solver::wrapper<BlockBackend>> CPRSolver;
    typedef amgcl::make_solver<amgcl::preconditioner::cpr_drs<PPrecond, SPrecond>, amgcl::runtime::solver::wrapper<BlockBackend>> CPRDRSSolver;

    const size_t rows = rhs.size();
    if (rows % B != 0) {
        PyErr_SetString(PyExc_ValueError, "RHS length is not divisible by block size");
        return nullptr;
    }
    if (ptr.size() != rows + 1) {
        PyErr_SetString(PyExc_ValueError, "CSR row pointer length does not match RHS length");
        return nullptr;
    }
    auto prm = make_cpr_prm(tolerance, max_iter, B, active_rows, solver, coarsening,
                            relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps,
                            aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                            ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping,
                            iluk_k, ilut_p, ilut_tau, gmres_m);
    boost::property_tree::ptree backend_prm;
    auto As = std::tie(rows, ptr, col, val);
    auto Ab = amgcl::adapter::block_matrix<val_type>(As);

    const rhs_type *rhs_ptr = reinterpret_cast<const rhs_type *>(rhs.data());
    std::vector<rhs_type> x(rows / B, amgcl::math::zero<rhs_type>());
    size_t iters = 0;
    double error = 0.0;
    auto start = std::chrono::steady_clock::now();
    if (use_drs) {
        CPRDRSSolver solve(Ab, prm, backend_prm);
        std::tie(iters, error) = solve(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + rows / B), x);
    } else {
        CPRSolver solve(Ab, prm, backend_prm);
        std::tie(iters, error) = solve(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + rows / B), x);
    }
    auto stop = std::chrono::steady_clock::now();

    PyObject *list = PyList_New(static_cast<Py_ssize_t>(rows));
    if (!list) return nullptr;
    for (size_t i = 0; i < rows / B; ++i) {
        for (int j = 0; j < B; ++j) {
            PyObject *v = PyFloat_FromDouble(x[i](j));
            if (!v) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i * B + j), v);
        }
    }
    double elapsed = std::chrono::duration<double>(stop - start).count();
    PyObject *ret = PyTuple_New(4);
    if (!ret) {
        Py_DECREF(list);
        return nullptr;
    }
    PyTuple_SET_ITEM(ret, 0, list);
    PyTuple_SET_ITEM(ret, 1, PyLong_FromLong(static_cast<long>(iters)));
    PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(error));
    PyTuple_SET_ITEM(ret, 3, PyFloat_FromDouble(elapsed));
    return ret;
}

static PyObject *solve_block_cpr(PyObject *, PyObject *args) {
    if (PyTuple_Size(args) != 30) {
        PyErr_SetString(PyExc_TypeError, "solve_block_cpr expects 30 positional arguments");
        return nullptr;
    }
    PyObject *ptr_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *col_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *val_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *rhs_obj = PyTuple_GET_ITEM(args, 3);
    int block_size = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 4)));
    double tolerance = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 5));
    int max_iter = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 6)));
    int active_rows = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 7)));
    const char *solver_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 8));
    const char *coarsening_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 9));
    const char *relaxation_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 10));
    const char *s_relaxation_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 11));
    bool use_drs = PyObject_IsTrue(PyTuple_GET_ITEM(args, 12)) != 0;
    double drs_eps_dd = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 13));
    double drs_eps_ps = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 14));
    double aggr_eps_strong = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 15));
    double aggr_over_interp = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 16));
    double aggr_relax = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 17));
    int npre = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 18)));
    int npost = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 19)));
    int ncycle = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 20)));
    bool direct_coarse = PyObject_IsTrue(PyTuple_GET_ITEM(args, 21)) != 0;
    int coarse_enough = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 22)));
    int max_levels = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 23)));
    double ilu_damping = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 24));
    int iluk_k = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 25)));
    int ilut_p = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 26)));
    double ilut_tau = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 27));
    int gmres_m = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 28)));
    int reserved = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 29)));
    (void)reserved;
    if (PyErr_Occurred() || !solver_c || !coarsening_c || !relaxation_c || !s_relaxation_c) {
        return nullptr;
    }

    std::vector<int> ptr, col;
    std::vector<double> val, rhs;
    if (!copy_buffer<int>(ptr_obj, ptr, "ptr") ||
        !copy_buffer<int>(col_obj, col, "col") ||
        !copy_buffer<double>(val_obj, val, "val") ||
        !copy_buffer<double>(rhs_obj, rhs, "rhs")) {
        return nullptr;
    }

    try {
        switch (block_size) {
        case 2:
            return solve_block_cpr_t<2>(ptr, col, val, rhs, tolerance, max_iter, active_rows,
                solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd,
                drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k,
                ilut_p, ilut_tau, gmres_m);
        case 3:
            return solve_block_cpr_t<3>(ptr, col, val, rhs, tolerance, max_iter, active_rows,
                solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd,
                drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k,
                ilut_p, ilut_tau, gmres_m);
        case 4:
            return solve_block_cpr_t<4>(ptr, col, val, rhs, tolerance, max_iter, active_rows,
                solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd,
                drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k,
                ilut_p, ilut_tau, gmres_m);
        case 5:
            return solve_block_cpr_t<5>(ptr, col, val, rhs, tolerance, max_iter, active_rows,
                solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd,
                drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k,
                ilut_p, ilut_tau, gmres_m);
        default:
            PyErr_SetString(PyExc_ValueError, "Only block sizes 2, 3, 4 and 5 are enabled");
            return nullptr;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown AMGCL block CPR exception");
        return nullptr;
    }
}

template <int B>
static PyObject *solve_bcsr_block_cpr_t(
    const std::vector<int> &ptr_i,
    const std::vector<int> &col_i,
    const std::vector<double> &val,
    const std::vector<double> &rhs,
    double tolerance, int max_iter, int active_rows,
    const std::string &solver, const std::string &coarsening,
    const std::string &relaxation, const std::string &s_relaxation,
    bool use_drs, double drs_eps_dd, double drs_eps_ps,
    double aggr_eps_strong, double aggr_over_interp, double aggr_relax,
    int npre, int npost, int ncycle, bool direct_coarse,
    int coarse_enough, int max_levels, double ilu_damping,
    int iluk_k, int ilut_p, double ilut_tau, int gmres_m,
    int reserved
) {
    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;
    typedef amgcl::backend::builtin<val_type> BlockBackend;
    typedef amgcl::backend::builtin<double> ScalarBackend;
    typedef amgcl::amg<ScalarBackend, amgcl::runtime::coarsening::wrapper, amgcl::runtime::relaxation::wrapper> PPrecond;
    typedef amgcl::relaxation::as_preconditioner<BlockBackend, amgcl::runtime::relaxation::wrapper> SPrecond;
    typedef amgcl::make_solver<amgcl::preconditioner::cpr<PPrecond, SPrecond>, amgcl::runtime::solver::wrapper<BlockBackend>> CPRSolver;
    typedef amgcl::make_solver<amgcl::preconditioner::cpr_drs<PPrecond, SPrecond>, amgcl::runtime::solver::wrapper<BlockBackend>> CPRDRSSolver;
    static std::shared_ptr<CPRSolver> cpr_solver;
    static std::shared_ptr<CPRDRSSolver> cpr_drs_solver;
    static size_t cached_cpr_rows = 0;
    static size_t cached_drs_rows = 0;

    const size_t nblocks = rhs.size() / B;
    if (rhs.size() % B != 0) {
        PyErr_SetString(PyExc_ValueError, "RHS length is not divisible by block size");
        return nullptr;
    }
    if (ptr_i.size() != nblocks + 1) {
        PyErr_SetString(PyExc_ValueError, "BCSR row pointer length does not match block row count");
        return nullptr;
    }
    if (val.size() != static_cast<size_t>(ptr_i.back()) * B * B) {
        PyErr_SetString(PyExc_ValueError, "BCSR value length must equal nnz_blocks * block_size^2");
        return nullptr;
    }
    std::vector<ptrdiff_t> ptr(ptr_i.begin(), ptr_i.end());
    std::vector<ptrdiff_t> col(col_i.begin(), col_i.end());

    auto prm = make_cpr_prm(tolerance, max_iter, B, active_rows, solver, coarsening,
                            relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps,
                            aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost,
                            ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping,
                            iluk_k, ilut_p, ilut_tau, gmres_m);
    boost::property_tree::ptree backend_prm;
    const val_type *v_ptr = reinterpret_cast<const val_type *>(val.data());
    auto matrix = amgcl::adapter::zero_copy(nblocks, ptr.data(), col.data(), v_ptr);

    const rhs_type *rhs_ptr = reinterpret_cast<const rhs_type *>(rhs.data());
    std::vector<rhs_type> x(nblocks, amgcl::math::zero<rhs_type>());
    size_t iters = 0;
    double error = 0.0;
    bool use_reuse = (reserved & 1) != 0;
    bool update_sprecond = (reserved & 2) != 0;
    bool update_ptransfer = (reserved & 4) != 0;
    bool reset_cache = (reserved & 8) != 0;
    bool did_setup = true;
    double update_time = 0.0;
    auto start = std::chrono::steady_clock::now();
    if (!use_reuse) {
        if (use_drs) {
            CPRDRSSolver solve(matrix, prm, backend_prm);
            std::tie(iters, error) = solve(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        } else {
            CPRSolver solve(matrix, prm, backend_prm);
            std::tie(iters, error) = solve(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        }
        did_setup = true;
    } else if (use_drs) {
        if (reset_cache || (cpr_drs_solver && cached_drs_rows != nblocks)) {
            cpr_drs_solver.reset();
            cached_drs_rows = 0;
        }
        if (!cpr_drs_solver) {
            cpr_drs_solver = std::make_shared<CPRDRSSolver>(matrix, prm, backend_prm);
            cached_drs_rows = nblocks;
            did_setup = true;
            std::tie(iters, error) = (*cpr_drs_solver)(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        } else {
            did_setup = false;
            if (update_sprecond) {
                auto u0 = std::chrono::steady_clock::now();
                cpr_drs_solver->precond().partial_update(*matrix, update_ptransfer);
                auto u1 = std::chrono::steady_clock::now();
                update_time = std::chrono::duration<double>(u1 - u0).count();
            }
            std::tie(iters, error) = (*cpr_drs_solver)(*matrix, amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        }
    } else {
        if (reset_cache || (cpr_solver && cached_cpr_rows != nblocks)) {
            cpr_solver.reset();
            cached_cpr_rows = 0;
        }
        if (!cpr_solver) {
            cpr_solver = std::make_shared<CPRSolver>(matrix, prm, backend_prm);
            cached_cpr_rows = nblocks;
            did_setup = true;
            std::tie(iters, error) = (*cpr_solver)(amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        } else {
            did_setup = false;
            if (update_sprecond) {
                auto u0 = std::chrono::steady_clock::now();
                cpr_solver->precond().partial_update(*matrix, update_ptransfer);
                auto u1 = std::chrono::steady_clock::now();
                update_time = std::chrono::duration<double>(u1 - u0).count();
            }
            std::tie(iters, error) = (*cpr_solver)(*matrix, amgcl::make_iterator_range(rhs_ptr, rhs_ptr + nblocks), x);
        }
    }
    auto stop = std::chrono::steady_clock::now();

    PyObject *list = PyList_New(static_cast<Py_ssize_t>(rhs.size()));
    if (!list) return nullptr;
    for (size_t i = 0; i < nblocks; ++i) {
        for (int j = 0; j < B; ++j) {
            PyObject *v = PyFloat_FromDouble(x[i](j));
            if (!v) {
                Py_DECREF(list);
                return nullptr;
            }
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i * B + j), v);
        }
    }
    double elapsed = std::chrono::duration<double>(stop - start).count();
    PyObject *ret = PyTuple_New(6);
    if (!ret) {
        Py_DECREF(list);
        return nullptr;
    }
    PyTuple_SET_ITEM(ret, 0, list);
    PyTuple_SET_ITEM(ret, 1, PyLong_FromLong(static_cast<long>(iters)));
    PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(error));
    PyTuple_SET_ITEM(ret, 3, PyFloat_FromDouble(elapsed));
    PyTuple_SET_ITEM(ret, 4, PyBool_FromLong(did_setup ? 1 : 0));
    PyTuple_SET_ITEM(ret, 5, PyFloat_FromDouble(update_time));
    return ret;
}

static PyObject *solve_bcsr_block_cpr(PyObject *, PyObject *args) {
    if (PyTuple_Size(args) != 30) {
        PyErr_SetString(PyExc_TypeError, "solve_bcsr_block_cpr expects 30 positional arguments");
        return nullptr;
    }
    PyObject *ptr_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *col_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *val_obj = PyTuple_GET_ITEM(args, 2);
    PyObject *rhs_obj = PyTuple_GET_ITEM(args, 3);
    int block_size = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 4)));
    double tolerance = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 5));
    int max_iter = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 6)));
    int active_rows = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 7)));
    const char *solver_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 8));
    const char *coarsening_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 9));
    const char *relaxation_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 10));
    const char *s_relaxation_c = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 11));
    bool use_drs = PyObject_IsTrue(PyTuple_GET_ITEM(args, 12)) != 0;
    double drs_eps_dd = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 13));
    double drs_eps_ps = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 14));
    double aggr_eps_strong = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 15));
    double aggr_over_interp = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 16));
    double aggr_relax = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 17));
    int npre = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 18)));
    int npost = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 19)));
    int ncycle = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 20)));
    bool direct_coarse = PyObject_IsTrue(PyTuple_GET_ITEM(args, 21)) != 0;
    int coarse_enough = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 22)));
    int max_levels = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 23)));
    double ilu_damping = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 24));
    int iluk_k = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 25)));
    int ilut_p = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 26)));
    double ilut_tau = PyFloat_AsDouble(PyTuple_GET_ITEM(args, 27));
    int gmres_m = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 28)));
    int reserved = static_cast<int>(PyLong_AsLong(PyTuple_GET_ITEM(args, 29)));
    if (PyErr_Occurred() || !solver_c || !coarsening_c || !relaxation_c || !s_relaxation_c) {
        return nullptr;
    }
    std::vector<int> ptr, col;
    std::vector<double> val, rhs;
    if (!copy_buffer<int>(ptr_obj, ptr, "ptr") ||
        !copy_buffer<int>(col_obj, col, "col") ||
        !copy_buffer<double>(val_obj, val, "val") ||
        !copy_buffer<double>(rhs_obj, rhs, "rhs")) {
        return nullptr;
    }
    try {
        switch (block_size) {
        case 2:
            return solve_bcsr_block_cpr_t<2>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m, reserved);
        case 3:
            return solve_bcsr_block_cpr_t<3>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m, reserved);
        case 4:
            return solve_bcsr_block_cpr_t<4>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m, reserved);
        case 5:
            return solve_bcsr_block_cpr_t<5>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver_c, coarsening_c, relaxation_c, s_relaxation_c, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m, reserved);
        default:
            PyErr_SetString(PyExc_ValueError, "Only block sizes 2, 3, 4 and 5 are enabled");
            return nullptr;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown AMGCL block CPR exception");
        return nullptr;
    }
}

static PyMethodDef Methods[] = {
    {"solve_block_cpr", solve_block_cpr, METH_VARARGS, "Solve a cell-major scalar CSR system with AMGCL block CPR."},
    {"solve_bcsr_block_cpr", solve_bcsr_block_cpr, METH_VARARGS, "Solve a true BCSR system with AMGCL block CPR."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "pyamgcl_block_cpr_capi_ext",
    nullptr,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_pyamgcl_block_cpr_capi_ext(void) {
    return PyModule_Create(&Module);
}
