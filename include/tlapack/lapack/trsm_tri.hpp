/// @file mult_uhu.hpp
/// @author Ella Addison-Taylor, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_TRSM_TRI
#define TLAPACK_TRSM_TRI

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/trmm.hpp"
#include <iomanip>

namespace tlapack {
    
template <typename matrix_t>
void printMatrix2(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(11) << A(i, j) << " ";
    }
    std::cout << std::endl;
}

template <typename matrixA_t, typename matrixB_t>
void trsm_tri(Side sideA,
              Uplo uploB,
              Op transA,
              Diag diagA,
              const matrixA_t& A,
              matrixB_t& B)
{
    using T = tlapack::type_t<matrixB_t>;
    using idx_t = tlapack::size_type<matrixB_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    tlapack::Uplo uploA;
    if (transA != tlapack::Op::NoTrans)
        uploA = (uploB == tlapack::Uplo::Upper) ? tlapack::Uplo::Lower : tlapack::Uplo::Upper;
    else
        uploA = uploB;

    idx_t n = nrows(A);
    
    if (n == 1) {
        if (transA == tlapack::Op::ConjTrans) 
            B(0, 0) /= conj(A(0, 0));
        else
            B(0, 0) /= A(0, 0);
        return;
    }

    idx_t nd = n / 2;

    auto A00 = slice(A, range(0, nd), range(0, nd));
    auto A01 = slice(A, range(0, nd), range(nd, n));
    auto A10 = slice(A, range(nd, n), range(0, nd));
    auto A11 = slice(A, range(nd, n), range(nd, n));

    auto B00 = slice(B, range(0, nd), range(0, nd));
    auto B01 = slice(B, range(0, nd), range(nd, n));    
    auto B10 = slice(B, range(nd, n), range(0, nd));
    auto B11 = slice(B, range(nd, n), range(nd, n));

    // Create workspace
    tlapack::Create<matrixB_t> new_matrix;
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);
    auto work00 = slice(work, range(0, nd), range(0, nd));
    auto work11 = slice(work, range(nd, n), range(nd, n));


    if (sideA == tlapack::Side::Left) {
        if (transA == tlapack::Op::NoTrans) {
            // Form: B := alpha*inv(A)*B
            if (uploB == tlapack::Uplo::Upper) {

                // Left, NoTrans, Upper, Diag
                trsm_tri(sideA, uploB, transA, diagA, A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                idx_t wm = nrows(B11);
                idx_t wn = ncols(B11);
                for (idx_t j = 0; j < wn; ++j)
                    for (idx_t i = 0; i < j + 1; ++i)
                        work11(i, j) = B11(i, j);

                gemm(transA, Op::NoTrans, real_t(-1), A01, work11, real_t(1), B01);

                // trmm_out(Side::Right, uplo, Op::NoTrans, diagA,
                //          transA, real_t(-1), B11, A01, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B01);

                return;
            }
            else {
                // Left, NoTrans, Lower, Diag
                trsm_tri(sideA, uploB, transA, diagA, A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                idx_t wm = nrows(B00);
                idx_t wn = ncols(B00);
                for (idx_t j = 0; j < wn; ++j)
                    for (idx_t i = j; i < wm; ++i)
                        work00(i, j) = B00(i, j);

                gemm(transA, Op::NoTrans, real_t(-1), A10, work00, real_t(1), B10);

                // trmm_out(Side::Right, uplo, Op::NoTrans, Diag::NonUnit, transA,
                //          real_t(-1), B00, A10, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B10);

                return;
            }
        }
        else {
            // Form: B := alpha*inv(A**T)*B
            // Form: B := alpha*inv(A**H)*B
            if (uploB == tlapack::Uplo::Upper) {
                // Left, Trans or ConjTrans, Upper, Diag

                trsm_tri(sideA, uploB, transA, diagA, A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                idx_t wm = nrows(B11);
                idx_t wn = ncols(B11);
                for (idx_t j = 0; j < wn; ++j)
                    for (idx_t i = 0; i < j + 1; ++i)
                        work11(i, j) = B11(i, j);

                gemm(transA, Op::NoTrans, real_t(-1), A10, work11, real_t(1), B01);
                //gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A01, B11, real_t(1), B01);

                // trmm_out(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                //          transA, real_t(-1), B11, A10, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B01);
                
                return;
            }
            else {
                // Left, Trans or ConjTrans, Lower, Diag

                trsm_tri(sideA, uploB, transA, diagA, A00, B00);
                
                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                idx_t wm = nrows(B00);
                idx_t wn = ncols(B00);
                for (idx_t j = 0; j < wn; ++j)
                    for (idx_t i = j; i < wm; ++i)
                        work00(i, j) = B00(i, j);

                gemm(transA, Op::NoTrans, real_t(-1), A01, work00, real_t(1), B10);
                //gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A10, B00, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B10);
                return;
            }
        }
    }
    else {
        if (transA == tlapack::Op::NoTrans) {
            // Form: B := alpha*B*inv(A)
            if (uploB == tlapack::Uplo::Upper) {
                // Right, NoTrans, Upper, Diag

                trsm_tri(sideA, uploB, transA, diagA, A00, B00);
                
                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                gemm(Op::NoTrans, Op::NoTrans, real_t(-1), B00, A01, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B01);
                
                return;
            }
            else {
                // Right, NoTrans, Lower, Diag
                trsm_tri(sideA, uploB, transA, diagA, A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                gemm(Op::NoTrans, Op::NoTrans, real_t(-1), B11, A10, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B10);

                return;
            }
        }
        else {
            // Form: B := alpha*B*inv(A**T)
            // Form: B := alpha*B*inv(A**H)

            if (uploB == tlapack::Uplo::Upper) {
                // Right, Trans or ConjTrans, Upper, Diag
                trsm_tri(sideA, uploB, transA, diagA, A00, B00);
                
                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                gemm(Op::NoTrans, Op::NoTrans, real_t(-1), B00, A01, real_t(1), B01);

                trsm(sideA, uploA, transA, diagA, real_t(1), A11, B01);
                
                return;
            }
            else {
                // Right, Trans or ConjTrans, Lower, Diag

                trsm_tri(sideA, uploB, transA, diagA, A00, B00);

                trsm_tri(sideA, uploB, transA, diagA, A11, B11);

                gemm(Op::NoTrans, Op::NoTrans, real_t(-1), B11, A10, real_t(1), B10);

                trsm(sideA, uploA, transA, diagA, real_t(1), A00, B10);
                return;
            }
        }
    }
}
}  // namespace tlapack

#endif  // TLAPACK_TRSM_TRI