#include <chrono>
#include <cstring>
#include <mutex>
#include <tuple>
#include <vector>
#include <string>

#include <boost/property_tree/ptree.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

namespace py = pybind11;

template <typename T>
std::vector<T> make_range(py::array_t<T, py::array::c_style | py::array::forcecast> a) {
    py::buffer_info i = a.request();
    if (i.ndim != 1) {
        throw std::runtime_error("Expected one-dimensional array");
    }
    const T *first = static_cast<const T *>(i.ptr);
    return std::vector<T>(first, first + i.shape[0]);
}

static boost::property_tree::ptree make_cpr_prm(
    double tolerance,
    int max_iter,
    int block_size,
    int active_rows,
    const std::string &solver,
    const std::string &coarsening,
    const std::string &relaxation,
    const std::string &s_relaxation,
    bool use_drs,
    double drs_eps_dd,
    double drs_eps_ps,
    double aggr_eps_strong,
    double aggr_over_interp,
    double aggr_relax,
    int npre,
    int npost,
    int ncycle,
    bool direct_coarse,
    int coarse_enough,
    int max_levels,
    double ilu_damping,
    int iluk_k,
    int ilut_p,
    double ilut_tau,
    int gmres_m
) {
    boost::property_tree::ptree prm;

    prm.put("solver.type", solver);
    prm.put("solver.tol", tolerance);
    prm.put("solver.check_after", true);
    if (max_iter > 0) {
        prm.put("solver.maxiter", max_iter);
    }
    if (gmres_m > 0) {
        prm.put("solver.M", gmres_m);
    }

    prm.put("precond.block_size", block_size);
    prm.put("precond.active_rows", active_rows);

    prm.put("precond.pprecond.coarsening.type", coarsening);
    prm.put("precond.pprecond.coarsening.aggr.eps_strong", aggr_eps_strong);
    prm.put("precond.pprecond.coarsening.over_interp", aggr_over_interp);
    prm.put("precond.pprecond.coarsening.relax", aggr_relax);
    prm.put("precond.pprecond.direct_coarse", direct_coarse);
    if (coarse_enough >= 0) {
        prm.put("precond.pprecond.coarse_enough", coarse_enough);
    }
    if (max_levels >= 0) {
        prm.put("precond.pprecond.max_levels", max_levels);
    }
    if (npre >= 0) {
        prm.put("precond.pprecond.npre", npre);
    }
    if (npost >= 0) {
        prm.put("precond.pprecond.npost", npost);
    }
    if (ncycle >= 0) {
        prm.put("precond.pprecond.ncycle", ncycle);
    }

    prm.put("precond.pprecond.relax.type", relaxation);
    prm.put("precond.pprecond.relax.damping", ilu_damping);
    prm.put("precond.pprecond.relax.k", iluk_k);
    prm.put("precond.pprecond.relax.p", ilut_p);
    prm.put("precond.pprecond.relax.tau", ilut_tau);

    prm.put("precond.sprecond.type", s_relaxation);
    prm.put("precond.sprecond.damping", ilu_damping);
    prm.put("precond.sprecond.k", iluk_k);
    prm.put("precond.sprecond.p", ilut_p);
    prm.put("precond.sprecond.tau", ilut_tau);

    if (use_drs) {
        prm.put("precond.eps_dd", drs_eps_dd);
        prm.put("precond.eps_ps", drs_eps_ps);
    }
    return prm;
}

template <int B>
py::tuple solve_block_cpr_impl(
    py::array_t<int, py::array::c_style | py::array::forcecast> ptr_a,
    py::array_t<int, py::array::c_style | py::array::forcecast> col_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> val_a,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_a,
    double tolerance,
    int max_iter,
    int active_rows,
    const std::string &solver,
    const std::string &coarsening,
    const std::string &relaxation,
    const std::string &s_relaxation,
    bool use_drs,
    double drs_eps_dd,
    double drs_eps_ps,
    double aggr_eps_strong,
    double aggr_over_interp,
    double aggr_relax,
    int npre,
    int npost,
    int ncycle,
    bool direct_coarse,
    int coarse_enough,
    int max_levels,
    double ilu_damping,
    int iluk_k,
    int ilut_p,
    double ilut_tau,
    int gmres_m
) {
    typedef amgcl::static_matrix<double, B, B> val_type;
    typedef amgcl::static_matrix<double, B, 1> rhs_type;
    typedef amgcl::backend::builtin<val_type> BlockBackend;
    typedef amgcl::backend::builtin<double> ScalarBackend;
    typedef amgcl::amg<
        ScalarBackend,
        amgcl::runtime::coarsening::wrapper,
        amgcl::runtime::relaxation::wrapper
    > PPrecond;
    typedef amgcl::relaxation::as_preconditioner<
        BlockBackend,
        amgcl::runtime::relaxation::wrapper
    > SPrecond;

    std::vector<int> ptr = make_range<int>(ptr_a);
    std::vector<int> col = make_range<int>(col_a);
    std::vector<double> val = make_range<double>(val_a);
    std::vector<double> rhs = make_range<double>(rhs_a);

    const size_t rows = rhs.size();
    if (rows % B != 0) {
        throw std::runtime_error("RHS length is not divisible by block size");
    }
    if (ptr.size() != rows + 1) {
        throw std::runtime_error("CSR row pointer length does not match RHS length");
    }

    auto prm = make_cpr_prm(
        tolerance, max_iter, B, active_rows,
        solver, coarsening, relaxation, s_relaxation,
        use_drs, drs_eps_dd, drs_eps_ps,
        aggr_eps_strong, aggr_over_interp, aggr_relax,
        npre, npost, ncycle, direct_coarse, coarse_enough, max_levels,
        ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m
    );

    auto As = std::tie(rows, ptr, col, val);
    auto Ab = amgcl::adapter::block_matrix<val_type>(As);

    boost::property_tree::ptree backend_prm;
    typedef amgcl::make_solver<
        amgcl::preconditioner::cpr<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<BlockBackend>
    > CPRSolver;
    typedef amgcl::make_solver<
        amgcl::preconditioner::cpr_drs<PPrecond, SPrecond>,
        amgcl::runtime::solver::wrapper<BlockBackend>
    > CPRDRSSolver;

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

    std::vector<double> out(rows, 0.0);
    for (size_t i = 0; i < rows / B; ++i) {
        for (int j = 0; j < B; ++j) {
            out[i * B + j] = x[i](j);
        }
    }
    double elapsed = std::chrono::duration<double>(stop - start).count();
    return py::make_tuple(py::array_t<double>(out.size(), out.data()), static_cast<int>(iters), error, elapsed);
}

py::tuple solve_block_cpr(
    py::array_t<int, py::array::c_style | py::array::forcecast> ptr,
    py::array_t<int, py::array::c_style | py::array::forcecast> col,
    py::array_t<double, py::array::c_style | py::array::forcecast> val,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs,
    int block_size,
    double tolerance,
    int max_iter,
    int active_rows,
    const std::string &solver,
    const std::string &coarsening,
    const std::string &relaxation,
    const std::string &s_relaxation,
    bool use_drs,
    double drs_eps_dd,
    double drs_eps_ps,
    double aggr_eps_strong,
    double aggr_over_interp,
    double aggr_relax,
    int npre,
    int npost,
    int ncycle,
    bool direct_coarse,
    int coarse_enough,
    int max_levels,
    double ilu_damping,
    int iluk_k,
    int ilut_p,
    double ilut_tau,
    int gmres_m
) {
    switch (block_size) {
    case 2:
        return solve_block_cpr_impl<2>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver, coarsening, relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m);
    case 3:
        return solve_block_cpr_impl<3>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver, coarsening, relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m);
    case 4:
        return solve_block_cpr_impl<4>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver, coarsening, relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m);
    case 5:
        return solve_block_cpr_impl<5>(ptr, col, val, rhs, tolerance, max_iter, active_rows, solver, coarsening, relaxation, s_relaxation, use_drs, drs_eps_dd, drs_eps_ps, aggr_eps_strong, aggr_over_interp, aggr_relax, npre, npost, ncycle, direct_coarse, coarse_enough, max_levels, ilu_damping, iluk_k, ilut_p, ilut_tau, gmres_m);
    default:
        throw std::runtime_error("Only block sizes 2, 3, 4 and 5 are enabled in pyamgcl_block_cpr");
    }
}

PYBIND11_MODULE(pyamgcl_block_cpr_ext, m) {
    m.def("solve_block_cpr", &solve_block_cpr,
        py::arg("ptr"),
        py::arg("col"),
        py::arg("val"),
        py::arg("rhs"),
        py::arg("block_size"),
        py::arg("tolerance") = 1e-6,
        py::arg("max_iter") = 100,
        py::arg("active_rows") = 0,
        py::arg("solver") = "bicgstab",
        py::arg("coarsening") = "aggregation",
        py::arg("relaxation") = "spai0",
        py::arg("s_relaxation") = "ilu0",
        py::arg("use_drs") = false,
        py::arg("drs_eps_dd") = 0.2,
        py::arg("drs_eps_ps") = 0.02,
        py::arg("aggr_eps_strong") = 0.08,
        py::arg("aggr_over_interp") = 1.0,
        py::arg("aggr_relax") = 2.0/3.0,
        py::arg("npre") = 1,
        py::arg("npost") = 1,
        py::arg("ncycle") = 1,
        py::arg("direct_coarse") = true,
        py::arg("coarse_enough") = -1,
        py::arg("max_levels") = -1,
        py::arg("ilu_damping") = 1.0,
        py::arg("iluk_k") = 1,
        py::arg("ilut_p") = 2,
        py::arg("ilut_tau") = 0.01,
        py::arg("gmres_m") = 30
    );
}
