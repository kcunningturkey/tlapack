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
#include <tlapack/LegacyBandedMatrix.hpp>

// <T>LAPACK
#include <tlapack/lapack/lanhe.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/pbtf0.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/potrf.hpp>
#include "pbtf0_fullaccess.hpp"
#include "pbtrf_fullaccess_slice_trapezoid.hpp"
#include "pbtrf_fullaccess.hpp"

// local file
#include "pbtrf_legacymatrix.hpp"

// C++ headers
#include <algorithm>

template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

template <typename matrix_t>
void printBandedMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);
    const idx_t kl = lowerband(A);
    const idx_t ku = upperband(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << ((i <= kl + j && j <= ku + i) ? A(i, j) : 0) << " ";
    }
}

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

    // tlapack::Uplo uplo = tlapack::Uplo::Lower;
    tlapack::Uplo uplo = tlapack::Uplo::Upper;

    std::vector<T> data1(m * n);
    for (int i = 0; i < m * n; ++i)
        data1[i] = i + 1;

    // Declacre matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);
    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);
    tlapack::LegacyBandedMatrix<T> TAB(m, m, 0, kd, &data1[0]);
    tlapack::LegacyBandedMatrix<T> TABl(m, m, kd, 0, &data1[0]);

    for (idx_t j = 0; j < n; j++) {
            real_t real_diag;   // Ensure diagonals are real
            real_diag = n * n;  // Strong positive diagonal
            TAB(j, j) = real_diag;
            TABl(j, j) = real_diag;

            if (uplo == tlapack::Uplo::Upper) {
        for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j; i++) {
                if constexpr (tlapack::is_complex<T>) {
                    TAB(i, j) =
                        T(static_cast<real_t>(i + 5),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    TAB(i, j) = static_cast<T>(i + 5);  // Only if T is real
                }
            }
    }
    else {
        for (idx_t i = j + 1; i < std::min(static_cast<int>(n),
                                               static_cast<int>(j + kd + 1));
                 i++) {
                if constexpr (tlapack::is_complex<T>) {
                    std::cout << "TAB(" << i << ", " << j << ")" << std::endl;
                    TABl(i, j) =
                        T(static_cast<real_t>(i + 5),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    TABl(i, j) = static_cast<T>(i + 5);  // Only if T is real
                }
            }
    }

            // for (idx_t i =
            //          std::max(0, static_cast<int>(j) - static_cast<int>(kd));
            //      i < j; i++) {
            //     if constexpr (tlapack::is_complex<T>) {
            //         TAB(i, j) =
            //             T(static_cast<real_t>(i + 5),
            //               static_cast<real_t>(j));  // Only if T is complex
            //     }
            //     else {
            //         TAB(i, j) = static_cast<T>(i + 5);  // Only if T is real
            //     }
            // }
        }

        // // printing banded
        // for (idx_t j = 0; j < n; j++) {
        //     std::cout << std::endl;
        //     for ( idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
        //          i < j; i++) {
        //             std::cout << TAB(i, j) << " ";
        //          }
        // }

    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>) {
                A(i, j) = T(static_cast<real_t>(0xCAFEBABE),
                            static_cast<real_t>(0xCAFEBABE));
            }
            else {
                A(i, j) = static_cast<real_t>(0xCAFEBABE);
            }
        }
    }

    for (idx_t j = 0; j < kd + 1; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>) {
                AB(j, i) = T(static_cast<real_t>(0xCAFEBABE),
                             static_cast<real_t>(0xCAFEBABE));
            }
            else {
                AB(j, i) = static_cast<real_t>(0xCAFEBABE);
            }
        }
    }
    // Create banded upper triangular matrix A
    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;   // Ensure diagonals are real
            real_diag = n * n + j;  // Strong positive diagonal
            A(j, j) = real_diag;

            for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j; i++) {
                if constexpr (tlapack::is_complex<T>) {
                    A(i, j) =
                        T(static_cast<real_t>(i + 5 + j),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    A(i, j) = static_cast<T>(i + 5 + j);  // Only if T is real
                }
            }
        }
    }
    else {  // tlapack::Uplo uplo = tlapack::Uplo::Lower;
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;   // Ensure diagonals are real
            real_diag = n * n + j;  // Strong positive diagonal
            A(j, j) = real_diag;

            for (idx_t i = j + 1; i < std::min(static_cast<int>(n),
                                               static_cast<int>(j + kd + 1));
                 i++) {
                if constexpr (tlapack::is_complex<T>) {
                    A(i, j) =
                        T(static_cast<real_t>(i + 5 + j),
                          static_cast<real_t>(j));  // Only if T is complex
                }
                else {
                    A(i, j) = static_cast<T>(i + 5 + j);  // Only if T is real
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
    
    // std::cout << "AB before = " << std::endl;   
    // printMatrix(AB);
    lacpy(tlapack::Uplo::General, A, A_copy);
    // potrf(uplo, A);

    tlapack::BlockedBandedFullCholeskyOpts opts;
    opts.nb = nb;

    // tlapack::BlockedBandedTestCholeskyOpts opts;
    // opts.nb = nb;
    
    // std::cout << "\nAB after = " << std::endl;
    pbtrf_fullaccess_slice_trapezoid(uplo, AB, opts);
    // pbtrf_fullaccess(uplo, A, kd, opts);
    // pbtf0(uplo, A);


    // printMatrix(AB);

    // std::cout << "starting pbtf0" << std::endl;
    // if (uplo == tlapack::Uplo::Upper) {
    // std::cout << "\nTAB after = " << std::endl;
    // pbtf0_fullaccess(uplo, TAB, kd);
    // printBandedMatrix(TAB);
    // }
    // else {
    //     std::cout << "\nTAB before = " << std::endl;
    //     printBandedMatrix(TABl);
    //     std::cout << "\nTAB after = " << std::endl;
    // pbtf0_fullaccess(uplo, TABl, kd);
    //     printBandedMatrix(TABl);
    // }

    // std::cout << "\nCorrect A = " << std::endl;
    // printMatrix(A);

    //-----------------------------------------------------------------------checking---------------------------------------------
    real_t normAbefore = lanhe(tlapack::Norm::Fro, uplo, A);

    // tlapack::BlockedBandedCholeskyOpts opts;
    // opts.nb = nb;

    // pbtrf_legacymatrix(uplo, AB, opts);
    // //pbtf0(uplo, AB);

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
    // std::cout << "AB into Acopy = " << std::endl;
    // printMatrix(A_copy);
    // else {
    //     for (idx_t j = 0; j < n; j++) {
    //         for (idx_t i = j; i < std::min(static_cast<int>(n),
    //                                        static_cast<int>(j + kd + 1));
    //              i++) {
    //             A_copy(i, j) = AB(i - j, j);
    //         }
    //     }
    //     mult_llh(A_copy);
    // }

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            A_copy(i, j) = A_copy(i, j) - A(i, j);
        }
    }

    // std::cout << "A_copy - A = " << std::endl;
    // printMatrix(A_copy);

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

    // GENERATE(1, 3, 10, 40, 130);
    // const idx_t kd = GENERATE(0, 1, 10, 20, 31);

    m = 11;
    n = m;
    kd = 3;
    nb = 2;

    // m = 69;
    // n = m;
    // kd = 31;
    // nb = 32;

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
