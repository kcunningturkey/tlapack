/// @file example_pbtrf.cpp
/// @author Kyle Cunningham, Henricus Bouwmeester, Ella Addison-Taylor
/// University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/LegacyBandedMatrix.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/potf2.hpp>
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK

// local file
#include "trmm_out.hpp"

// C++ headers
#include <algorithm>
#include <iomanip>

using namespace tlapack;

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(11) << A(i, j) << " ";
    }
    std::cout << std::endl;
}

template <typename uplo_t, typename matrix_t>
void trsm_tri(uplo_t uplo, matrix_t& A, matrix_t& B)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    idx_t n = nrows(A);

    tlapack::Create<matrix_t> new_matrix;

    if (n == 1) {
        B(0, 0) /= A(0, 0);
    }
    else {
        // Slices of A and B
        if (uplo == tlapack::Uplo::Lower) {

            idx_t nd = n / 2;

            auto A00 = slice(A, range(0, nd), range(0, nd));
            auto A10 = slice(A, range(nd, n), range(0, nd));
            auto A11 = slice(A, range(nd, n), range(nd, n));

            auto B00 = slice(B, range(0, nd), range(0, nd));
            auto B10 = slice(B, range(nd, n), range(0, nd));
            auto B11 = slice(B, range(nd, n), range(nd, n));

            trsm_tri(Uplo::Lower, A00, B00);

            trsm_tri(Uplo::Lower, A11, B11);

            // NEED TRMM_outofplace from Ella
            // gemm(Op::NoTrans, Op::NoTrans, real_t(-1), A10, B00, real_t(1),
            //      B10);
            
            trmm_out(Side::Right, Uplo::Lower, Op::NoTrans, Diag::NonUnit, Op::NoTrans, real_t(-1), B00, A10, real_t(1), B10);

            trsm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, real_t(1),
                 A11, B10);
        }
        else {
            return;
        }
    }
}
//------------------------------------------------------------------------------
template <typename T>
void run(size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    // tlapack::Uplo uplo = tlapack::Uplo::Upper;
    tlapack::Uplo uplo = tlapack::Uplo::Lower;

    tlapack::Create<matrix_t> new_matrix;

    // Create A
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);

    // Fill A with DEADBEEF
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if constexpr (tlapack::is_complex<T>) {
                A(i, j) = T(0, 0);
                // A(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                //             static_cast<real_t>(0xDEADBEEF));
            }
            else {
                A(i, j) = T(0);
                // A(i, j) = T(static_cast<real_t>(0xDEADBEEF));
            }
        }
    }

    // Fill the lower part of A
    if (uplo == tlapack::Uplo::Lower) {
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < i + 1; j++) {
                if constexpr (tlapack::is_complex<T>) {
                    A(i, j) = T(static_cast<real_t>(rand()) /
                                    static_cast<real_t>(RAND_MAX),
                                static_cast<real_t>(rand()) /
                                    static_cast<real_t>(RAND_MAX));
                }
                else {
                    A(i, j) = T(static_cast<real_t>(rand()) /
                                static_cast<real_t>(RAND_MAX));
                }
            }
        }
    }

    // Create B
    std::vector<T> B_;
    auto B = new_matrix(B_, n, n);

    // Fill B with DEADBEEF
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if constexpr (tlapack::is_complex<T>) {
                B(i, j) = T(0, 0);
                // B(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                //             static_cast<real_t>(0xDEADBEEF));
            }
            else {
                B(i, j) = T(0);
                // B(i, j) = T(static_cast<real_t>(0xDEADBEEF));
            }
        }
    }

    // Fill the lower part of B
    if (uplo == tlapack::Uplo::Lower) {
        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < i + 1; j++) {
                if constexpr (tlapack::is_complex<T>) {
                    B(i, j) = T(static_cast<real_t>(rand()) /
                                    static_cast<real_t>(RAND_MAX),
                                static_cast<real_t>(rand()) /
                                    static_cast<real_t>(RAND_MAX));
                }
                else {
                    B(i, j) = T(static_cast<real_t>(rand()) /
                                static_cast<real_t>(RAND_MAX));
                }
            }
        }
    }


    // Ensure the B is in the column space of A
    trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, real_t(1), A, B);

    std::vector<T> B_copy_;
    auto B_copy = new_matrix(B_copy_, n, n);
    lacpy(uplo, B, B_copy);

    real_t normB_orig = lange(Norm::Fro, B); 

    trsm_tri(uplo, A, B);
    
    // Check 
    trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, real_t(1), A, B);

    for (idx_t i = 0; i < n; i++){
        for (idx_t j = 0; j < n; j++){
            B(i,j) -= B_copy(i,j);
        }
    }
    
    real_t normB = lange(Norm::Fro, B); 

    std::cout << "Norm = " << normB/ normB_orig << std::endl;
    
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    idx_t m = 13;
    idx_t n = m;
    idx_t kd = 5;
    idx_t nb = 2;

    printf("run< float  >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    std::cout << std::endl;
    run<float>(n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<double>(n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<long double>(n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<float>>(n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<double>>(n);
    printf("-----------------------\n");

    return 0;
}