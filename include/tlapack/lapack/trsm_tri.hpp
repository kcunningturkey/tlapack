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

namespace tlapack {

template <typename matrixA_t, typename matrixB_t>
void trsm_tri(
    Side sideA, 
    Uplo uplo, 
    Op transA, 
    Diag diagA, 
    const matrixA_t& A, 
    matrixB_t& B)
{
    using T = tlapack::type_t<matrixB_t>;
    using idx_t = tlapack::size_type<matrixB_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    idx_t n = nrows(A);

    tlapack::Create<matrixB_t> new_matrix;

    if (transA == tlapack::Op::NoTrans) {
        if (n == 1) {
            B(0, 0) /= A(0, 0);
        }
        else {
            // Slices of A and B for Lower
            if (uplo == tlapack::Uplo::Lower) {
                idx_t nd = n / 2;

                auto A00 = slice(A, range(0, nd), range(0, nd));
                auto A10 = slice(A, range(nd, n), range(0, nd));
                auto A11 = slice(A, range(nd, n), range(nd, n));

                auto B00 = slice(B, range(0, nd), range(0, nd));
                auto B10 = slice(B, range(nd, n), range(0, nd));
                auto B11 = slice(B, range(nd, n), range(nd, n));

                trsm_tri(sideA, uplo, transA, diagA, A00, B00);

                trsm_tri(sideA, uplo, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                // gemm(transA, Op::NoTrans, real_t(-1), A10, B00,
                // real_t(1),
                //      B10);

                trmm_out(Side::Right, uplo, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A10, real_t(1), B10);

                trsm(sideA, uplo, transA, diagA, real_t(1), A11, B10);
            }
            else {  // tlapack::UPLO::Upper
                idx_t nd = n / 2;

                auto A00 = slice(A, range(0, nd), range(0, nd));
                auto A01 = slice(A, range(0, nd), range(nd, n));
                auto A11 = slice(A, range(nd, n), range(nd, n));

                auto B00 = slice(B, range(0, nd), range(0, nd));
                auto B01 = slice(B, range(0, nd), range(nd, n));
                auto B11 = slice(B, range(nd, n), range(nd, n));

                trsm_tri(sideA, uplo, transA, diagA, A11, B11);

                trsm_tri(sideA, uplo, transA, diagA, A00, B00);

                // NEED TRMM_outofplace from Ella
                // gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A01, B11,
                // real_t(1),
                //      B01);

                trmm_out(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                         Op::NoTrans, real_t(-1), B11, A01, real_t(1), B01);

                trsm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                     real_t(1), A00, B01);
            }
        }
    }

    else if (transA == tlapack::Op::Trans) {
        if (n == 1) {
            B(0, 0) /= A(0, 0);
        }
        else {
            // Slices of A and B for Lower
            if (uplo == tlapack::Uplo::Lower) {
                idx_t nd = n / 2;

                auto A00 = slice(A, range(0, nd), range(0, nd));
                auto A01 = slice(A, range(0, nd), range(nd, n));
                auto A11 = slice(A, range(nd, n), range(nd, n));

                auto B00 = slice(B, range(0, nd), range(0, nd));
                auto B10 = slice(B, range(nd, n), range(0, nd));
                auto B11 = slice(B, range(nd, n), range(nd, n));

                trsm_tri(sideA, uplo, transA, diagA, A00, B00);

                trsm_tri(sideA, uplo, transA, diagA, A11, B11);

                // NEED TRMM_outofplace from Ella
                // gemm(transA, Op::NoTrans, real_t(-1), A10, B00,
                // real_t(1),
                //      B10);

                trmm_out(Side::Right, uplo, Op::NoTrans, Diag::NonUnit, transA,
                         real_t(-1), B00, A01, real_t(1), B10);

                trsm(sideA, uplo, transA, diagA, real_t(1), A11, B10);
            }
            else {  // tlapack::UPLO::Upper
                idx_t nd = n / 2;

                auto A00 = slice(A, range(0, nd), range(0, nd));
                auto A01 = slice(A, range(0, nd), range(nd, n));
                auto A11 = slice(A, range(nd, n), range(nd, n));

                auto B00 = slice(B, range(0, nd), range(0, nd));
                auto B01 = slice(B, range(0, nd), range(nd, n));
                auto B11 = slice(B, range(nd, n), range(nd, n));

                trsm_tri(sideA, uplo, transA, diagA, A11, B11);

                trsm_tri(sideA, uplo, transA, diagA, A00, B00);

                // NEED TRMM_outofplace from Ella
                // gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A01, B11,
                // real_t(1),
                //      B01);

                trmm_out(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                         Op::NoTrans, real_t(-1), B11, A01, real_t(1), B01);

                trsm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
                     real_t(1), A00, B01);
            }
        }
    }
    else {
        std::cout << "ConjTrans" << std::endl;
        // ConjTrans
    }
}
}  // namespace tlapack

#endif  // TLAPACK_TRSM_TRI