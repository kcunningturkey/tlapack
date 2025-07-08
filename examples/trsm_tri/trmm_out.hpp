/// @file trmm_out.hpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRMM_OUT
#define TLAPACK_TRMM_OUT

// #include "../../../test/include/MatrixMarket.hpp"
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"

namespace tlapack {
/**
 *
 * @brief in-place multiplication of upper triangular matrix U and lower
 * triangular matrix U^H. This is the recursive variant.
 *
 * @param[in,out] U n-by-n matrix
 * On entry, the upper triangular matrix U. On exit, U contains the upper
 * part of the Hermitian product U^H*U. The lower triangular entries of U are
 * not referenced.
 *
 * @param[in] opts Options.
 *
 * @ingroup auxiliary
 */
template <TLAPACK_MATRIX matrixA_t,
          TLAPACK_MATRIX matrixB_t,
          TLAPACK_MATRIX matrixC_t,
          TLAPACK_SCALAR alpha_t,
          TLAPACK_SCALAR beta_t,
          class T = type_t<matrixB_t>,
          disable_if_allow_optblas_t<pair<matrixA_t, T>,
                                     pair<matrixB_t, T>,
                                     pair<matrixC_t, T>,
                                     pair<alpha_t, T>,
                                     pair<beta_t, T> > = 0>
void trmm_out(Side side,
              Uplo uplo,
              Op transA,
              Diag diag,
              Op transB,
              const alpha_t& alpha,
              const matrixA_t& A,
              const matrixB_t& B,
              const beta_t& beta,
              matrixC_t& C)
{
    using idx_t = tlapack::size_type<matrixA_t>;
    using range = pair<idx_t, idx_t>;
    using real_t = real_type<T>;

    if (transB == Op::NoTrans) {
        if (uplo == Uplo::Upper) {
            idx_t n = ncols(B);
            idx_t m = nrows(B);
            idx_t n0 = n / 2;
            if (n == 1) {
                for (idx_t i = 0; i < m; ++i) {
                    for (idx_t j = 0; j < n; ++j) {
                        C(i, j) = alpha * B(i, j) * A(0, 0) + beta * C(i, j);
                    }
                }
            }
            else {
                auto C0 = slice(C, range(0, m), range(0, n0));
                auto C1 = slice(C, range(0, m), range(n0, n));

                auto A00 = slice(A, range(0, n0), range(0, n0));
                auto A01 = slice(A, range(0, n0), range(n0, n));
                auto A11 = slice(A, range(n0, n), range(n0, n));

                auto B0 = slice(B, range(0, m), range(0, n0));
                auto B1 = slice(B, range(0, m), range(n0, n));

                trmm_out(side, uplo, transA, diag, transB, alpha, A00, B0, beta,
                         C0);

                gemm(Op::NoTrans, Op::NoTrans, alpha, B0, A01, beta, C1);

                trmm_out(side, uplo, transA, diag, transB, alpha, A11, B1,
                         real_t(1), C1);
            }
        }
        else {
            idx_t n = ncols(B);
            idx_t m = nrows(B);
            idx_t n0 = n / 2;
            if (n == 1) {
                for (idx_t i = 0; i < m; ++i) {
                    for (idx_t j = 0; j < n; ++j) {
                        C(i, j) = alpha * B(i, j) * A(0, 0) + beta * C(i, j);
                    }
                }
            }
            else {
                auto C0 = slice(C, range(0, m), range(0, n0));
                auto C1 = slice(C, range(0, m), range(n0, n));

                auto A00 = slice(A, range(0, n0), range(0, n0));
                auto A10 = slice(A, range(n0, n), range(0, n0));
                auto A11 = slice(A, range(n0, n), range(n0, n));

                auto B0 = slice(B, range(0, m), range(0, n0));
                auto B1 = slice(B, range(0, m), range(n0, n));

                trmm_out(side, uplo, transA, diag, transB, alpha, A11, B1, beta,
                         C1);

                gemm(Op::NoTrans, Op::NoTrans, alpha, B1, A10, beta, C0);

                trmm_out(side, uplo, transA, diag, transB, alpha, A00, B0,
                         real_t(1), C0);
            }
        }
    }
    else {
        idx_t n = nrows(B);
        idx_t m = ncols(B);
        idx_t n0 = n / 2;
        if (n == 1) {
            // We want to do AXPBY, as of today, this routine is not yet BLAS,
            // and moreover we need a conjugate on one of the B vector
            //
            // this is level 0 of an AXPBY with conjugate option
            for (idx_t i = 0; i < n; ++i) {
                for (idx_t j = 0; j < m; ++j) {
                    C(j, i) = alpha * conj(B(i, j)) * A(0, 0) + beta * C(j, i);
                }
            }
        }
        else {
            auto C0 = slice(C, range(0, m), range(0, n0));
            auto C1 = slice(C, range(0, m), range(n0, n));

            auto A00 = slice(A, range(0, n0), range(0, n0));
            auto A10 = slice(A, range(n0, n), range(0, n0));
            auto A11 = slice(A, range(n0, n), range(n0, n));

            auto B0 = slice(B, range(0, n0), range(0, m));
            auto B1 = slice(B, range(n0, n), range(0, m));

            trmm_out(side, uplo, transA, diag, transB, alpha, A11, B1, beta,
                     C1);

            gemm(Op::ConjTrans, Op::NoTrans, alpha, B1, A10, beta, C0);

            trmm_out(side, uplo, transA, diag, transB, alpha, A00, B0,
                     real_t(1), C0);
        }
    }
}

}  // namespace tlapack

#endif  // TLAPACK_TRMM_OUT
