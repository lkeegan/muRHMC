#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "dirac_op.hpp"
#include "inverters.hpp"
#include "io.hpp"
#include "rational_approx.hpp"
#include "rhmc.hpp"
#include "stats.hpp"

std::chrono::time_point<std::chrono::high_resolution_clock> timer_start;
// start timer
void tick() { timer_start = std::chrono::high_resolution_clock::now(); }
// stop timer and return elapsed time in milliseconds
int tock() {
  auto timer_stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(timer_stop -
                                                               timer_start)
      .count();
}

int main(int argc, char *argv[]) {
  // shifts for shifted solver
  std::vector<double> shifts = {0,    0,    1e-10, 1e-9, 1e-8, 1e-7, 3e-7,
                                1e-6, 3e-6, 1e-5,  3e-5, 1e-4, 3e-4, 1e-3,
                                2e-3, 3e-3, 5e-3,  1e-2, 3e-2, 1e-1, 3e-1,
                                1,    3,    1e2,   3e2,  1e3,  3e3,  1e4};
  int N_shifts = static_cast<int>(shifts.size());

  if (argc - 1 != 1) {
    std::cout << "Input file not specified, e.g." << std::endl;
    std::cout << "./measure input_file.txt" << std::endl;
    return 1;
  }

  std::cout.precision(14);

  rhmc_params rhmc_pars;
  run_params run_pars;

  read_input_file(argv[1], rhmc_pars, run_pars);
  rhmc_pars.n_pf = N_rhs;

  // make TxL^3 EO lattice
  lattice grid(run_pars.T, run_pars.L, true);

  double stopping_criterion = rhmc_pars.MD_eps;
  double stopping_criterion_shifts = 1.e-15;
  int max_iter = 1e9;

  log("");
  log("Shifted solver comparison with parameters:");
  log("");
  log("N_rhs", rhmc_pars.n_pf);
  log("mass", rhmc_pars.mass);
  log("base_name", run_pars.base_name);
  log("stopping_criterion", stopping_criterion);
  log("T", grid.L0);
  log("L1", grid.L1);
  log("L2", grid.L2);
  log("L3", grid.L3);
  log("config", run_pars.initial_config);

  rhmc hmc(rhmc_pars, grid);
  field<gauge> U(grid);
  dirac_op D(grid, rhmc_pars.mass, 0.0);
  read_gauge_field(U, run_pars.base_name, run_pars.initial_config);

  field<block_fermion> B(U.grid, field<block_fermion>::EVEN_ONLY);
  hmc.gaussian_fermion(B);

  // do SCG solve for each RHS of B separately
  std::vector<double> resSCG(N_shifts, 0.0);
  field<fermion> b(U.grid, field<fermion>::EVEN_ONLY), Ax(b);
  std::vector<field<fermion>> x(N_shifts, b);
  int iterSCG = 0;
  int timeSCG = 0;
  for (int i_rhs = 0; i_rhs < N_rhs; ++i_rhs) {
    // get i'th RHS column of block vector B
    for (int i_x = 0; i_x < B.V; ++i_x) {
      b[i_x] = B[i_x].col(i_rhs);
    }
    // do SCG solve
    tick();
    iterSCG += cg_multishift(x, b, U, shifts, D, stopping_criterion,
                             stopping_criterion_shifts, max_iter);
    timeSCG += tock();
    // measure residuals
    for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
      double shift = shifts[i_shift];
      D.DDdagger(Ax, x[i_shift], U);
      Ax.add(x[i_shift], shift);
      Ax -= b;
      double residual = sqrt(Ax.dot(Ax).real() / b.dot(b).real());
      if (residual > resSCG[i_shift]) {
        resSCG[i_shift] = residual;
      }
    }
  }

  // do SBCGrQ solve for B
  field<block_fermion> AX(B);
  std::vector<double> resSBCGrQ(N_shifts, 0.0);
  std::vector<field<block_fermion>> X(N_shifts, B);
  tick();
  int iterSBCGrQ = N_rhs * SBCGrQ(X, B, U, shifts, D, stopping_criterion,
                                  stopping_criterion_shifts, max_iter);
  int timeSBCGrQ = tock();
  block_matrix b2;
  hermitian_dot(B, B, b2);
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    double shift = shifts[i_shift];
    D.DDdagger(AX, X[i_shift], U);
    AX.add(X[i_shift], shift);
    AX -= B;
    block_matrix r2;
    hermitian_dot(AX, AX, r2);
    resSBCGrQ[i_shift] =
        sqrt((r2.diagonal().real().array() / b2.diagonal().array().real())
                 .maxCoeff());
  }

  std::vector<double> resSBCGrQ_old(N_shifts, 0.0);
  tick();
  int iterSBCGrQ_old =
      N_rhs * SBCGrQ_old(X, B, U, shifts, D, stopping_criterion,
                         stopping_criterion_shifts, max_iter);
  int timeSBCGrQ_old = tock();
  hermitian_dot(B, B, b2);
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    double shift = shifts[i_shift];
    D.DDdagger(AX, X[i_shift], U);
    AX.add(X[i_shift], shift);
    AX -= B;
    block_matrix r2;
    hermitian_dot(AX, AX, r2);
    resSBCGrQ_old[i_shift] =
        sqrt((r2.diagonal().real().array() / b2.diagonal().array().real())
                 .maxCoeff());
  }

  std::cout << "# Shift\t\t\tSCG\t\t\tSBCGrQ\t\t\tSBCGrQ_old" << std::endl;
  for (int i_shift = 0; i_shift < N_shifts; ++i_shift) {
    std::cout << "# " << std::scientific << shifts[i_shift] << "\t"
              << resSCG[i_shift] << "\t" << resSBCGrQ[i_shift] << "\t"
              << resSBCGrQ_old[i_shift] << std::endl;
  }

  std::cout << "# N\tSCG iter/time\t\tSBCGrQ iter/time\tSBCGrQ_old iter/time"
            << std::endl;
  std::cout << "  " << N_rhs << "\t";
  std::cout << iterSCG << "\t" << timeSCG << "\t\t";
  std::cout << iterSBCGrQ << "\t" << timeSBCGrQ << "\t\t";
  std::cout << iterSBCGrQ_old << "\t" << timeSBCGrQ_old << std::endl;

  return 0;
}