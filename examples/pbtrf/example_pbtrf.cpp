/// @file example_pbtrf.cpp
/// @author L. Carlos Gutierrez, Kyle Cunningham, Henricus Bouwmeester, Ella Addison-Taylor
/// University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/pbtf0.hpp>

// local file
#include "pbtrf.hpp"

// C++ headers
#include <algorithm>

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t kd, size_t nb)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Define parameters for banded and consolidated matrices

    tlapack::Uplo uplo = tlapack::Uplo::Upper;

    // Declacre matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>) {
                A(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                            static_cast<real_t>(0xDEADBEEF));
            }
            else {
                A(i, j) = static_cast<real_t>(0xDEADBEEF);
            }
        }
    }

    for (idx_t j = 0; j < kd + 1; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>) {
                AB(j, i) = T(static_cast<real_t>(0xDEADBEEF),
                             static_cast<real_t>(0xDEADBEEF));
            }
            else {
                AB(j, i) = static_cast<real_t>(0xDEADBEEF);
            }
        }
    }
    // Create banded upper triangular matrix A
    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;   // Ensure diagonals are real
            real_diag = n * n;  // Strong positive diagonal
            A(j, j) = real_diag;

            for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j; i++) {
                if constexpr (tlapack::is_complex<T>) {
                    A(i, j) =
                        T(static_cast<real_t>(i + 5),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    A(i, j) = static_cast<T>(i + 5);  // Only if T is real
                }
            }
        }
    }
    else {  // tlapack::Uplo uplo = tlapack::Uplo::Lower;
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;   // Ensure diagonals are real
            real_diag = n * n;  // Strong positive diagonal
            A(j, j) = real_diag;

            for (idx_t i = j + 1; i < std::min(static_cast<int>(n),
                                               static_cast<int>(j + kd + 1));
                 i++) {
                if constexpr (tlapack::is_complex<T>) {
                    A(i, j) =
                        T(static_cast<real_t>(i + 5),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    A(i, j) = static_cast<T>(i + 5);  // Only if T is real
                }
            }
        }
    }

    // Create matrix AB depending on if A is upper or lower
    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j + 1; i++) {
                AB(i + kd - j, j) = A(i, j);
            }
        }
    }
    else {  // tlapack::Uplo uplo = tlapack::Uplo::Lower;
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j; i < std::min(static_cast<int>(n),
                                           static_cast<int>(j + kd + 1));
                 i++) {
                AB(i - j, j) = A(i, j);
            }
        }
    }

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = kd; j < n; j++) {
            for (idx_t i = 0; i < j - kd; i++) {
                A(i, j) = static_cast<float>(0);
            }
        }
    }
    else {
        for (idx_t j = 0; j < n - kd; j++) {
            for (idx_t i = kd + j + 1; i < n; i++) {
                A(i, j) = static_cast<float>(0);
            }
        }
    }
    //-----------------------------------------------------------------------checking---------------------------------------------
    real_t normAbefore = lanhe(tlapack::Norm::Fro, uplo, A);

    tlapack::BlockedBandedCholeskyOpts opts;
    opts.nb = nb;

    pbtrf(uplo, AB, opts);

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j + 1; i++) {
                A_copy(i, j) = AB(i + kd - j, j);
            }
        }
        mult_uhu(A_copy);
    }
    else {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j; i < std::min(static_cast<int>(n),
                                           static_cast<int>(j + kd + 1));
                 i++) {
                A_copy(i, j) = AB(i - j, j);
            }
        }
        mult_llh(A_copy);
    }

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            A_copy(i, j) = A_copy(i, j) - A(i, j);
        }
    }

    real_t normA = lanhe(tlapack::Norm::Fro, uplo, A_copy);

    std::cout << "The norm is = " << normA / normAbefore << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << std::endl
              << "example_pbtrf executed" << std::endl
              << std::endl;

    using std::size_t;

    using idx_t = size_t;

    idx_t m, n, kd, nb;

    m = 130;
    n = m;
    kd = 31;
    nb = 20;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    run<float>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    run<double>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", static_cast<int>(m),
           static_cast<int>(n));
    run<long double>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<float>>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<double>>(m, n, kd, nb);
    printf("-----------------------\n");

    return 0;
}
