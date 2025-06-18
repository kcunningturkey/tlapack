/// @file example_pbtrf.cpp
/// @author L. Carlos Gutierrez, Kyle Cunningham, Henricus Bouwmeester,
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
#include <tlapack/lapack/pbtf2.hpp>
#include <tlapack/lapack/pbtrf.hpp>
#include <tlapack/lapack/potrf.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/mult_llh.hpp>
// #include "tlapack/base/utils.hpp"

// C++ headers
#include <algorithm>

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
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

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Define parameters for banded and consolidated matrices

    std::size_t kd = 7;
    tlapack::Uplo uplo = tlapack::Uplo::Upper;

    // Declacre matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> blAH_;
    auto blAH = new_matrix(blAH_, kd + 1, n);
    // std::vector<T> blAH_;
    // auto blAH = new_matrix(blAH_, m, n);
    std::vector<T> blAH2_;
    auto blAH2 = new_matrix(blAH2_, m, n);
    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    for (idx_t j = 0; j <n; ++j) {
        for (idx_t i = 0; i < n; ++i){
            if constexpr (tlapack::is_complex<T>) {
                A(i, j) = T(static_cast<real_t>(0xDEADBEEF), static_cast<real_t>(0xDEADBEEF));
            }
            else {
                A(i, j) = static_cast<real_t>(0xDEADBEEF);
            }
        }
    }
    // Create banded upper triangular matrix A
    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;        // Ensure diagonals are real
            real_diag = j * j + n + 10;  // Strong positive diagonal
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
    else { // tlapack::Uplo uplo = tlapack::Uplo::Lower;
        for (idx_t j = 0; j < n; j++) {
            real_t real_diag;        // Ensure diagonals are real
            real_diag = j * j + 10;  // Strong positive diagonal
            A(j, j) = real_diag;

            for (idx_t i = j + 1; i < std::min(static_cast<int>(n), static_cast<int>(j + kd + 1)); i++) {
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
    else{ // tlapack::Uplo uplo = tlapack::Uplo::Lower;
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j;
                 i < std::min(static_cast<int>(n), static_cast<int>(j + kd + 1)); i++) {
                AB(i - j, j) = A(i, j);
            }
        }
    }    

    // std::cout << std::endl << "AB before = ";
    // printMatrix(AB);  

    std::cout << std::endl << "A before = ";
    printMatrix(A);

    real_t normA = lange(tlapack::FROB_NORM, A);
    lacpy(tlapack::Uplo::General, AB, blAH);

    lacpy(tlapack::Uplo::General, A, blAH2);
    
    // std::cout << "blAH" << std::endl;
    // printMatrix(blAH);

    // std::cout << "blAH2" << std::endl;
    // printMatrix(blAH2);

    pbtrf(uplo, A, kd);

    pbtf2(uplo, blAH);

    // std::cout << "\npbtrf" << std::endl;
    // printMatrix(A);

    // std::cout << "\nlevel 0 factor" << std::endl;
    // printMatrix(blAH);

    mult_uhu(A);

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++){
            blAH2(i, j) = blAH2(i,j) - A(i, j);
        }
    } 

    real_t normB = lange(tlapack::FROB_NORM, blAH2);

    std::cout << "\nThe norm is " << normB/normA << std::endl;
    
    //----------------------------------------------------------level0---------------------------------------------
    // std::cout << std::endl << "A before = ";
    // printMatrix(A);

    // pbtf2(uplo, AB);
    // std::cout << "\npbtrf = " << std::endl;
    // printMatrix(AB);

    // real_t normPotrf = lange(tlapack::FROB_NORM, A);

    // // potrf(uplo, A);
    // // std::cout << "\npotrf = " << std::endl;
    // // printMatrix(A);

    // if (uplo == tlapack::Uplo::Upper) {
    //     for (idx_t j = 0; j < n; j++) {
    //         for (idx_t i = std::max(static_cast<int>(0), static_cast<int>(j - kd)); i < j+1; i++){
    //             blAH(i, j) = AB(i + kd -j, j);
    //         }
    //     }
    // lacpy(tlapack::Uplo::Upper, blAH, blAH2);
    // std::cout << std::endl << "blAH = ";
    // printMatrix(blAH);

    // mult_uhu(blAH2);
    // }
    // else {
    //     for (idx_t j = 0; j < n; j++) {
    //         for (idx_t i = 0; i < std::min(static_cast<int>(kd + 1), static_cast<int>(n-j)); i++) {
    //             //if (i + j < n)
    //                 blAH(i + j, j) = AB(i, j);
    //         }
    //         // for (idx_t i = 0; i < std::min(static_cast<int>(n), static_cast<int>(j + kd + 1)); i++) {
    //         //     blAH(i, j) = AB(i - j, j)
    //         // }
    //     }
    //     lacpy(tlapack::Uplo::Lower, blAH, blAH2);
    //     std::cout << std::endl << "blAH = ";
    //     printMatrix(blAH);

    //     mult_llh(blAH2);
    // }


    // // Print verbose
    // if (verbose) {
    //     // std::cout << std::endl << "potrf = ";
    //     // printMatrix(A);
    //     // std::cout << std::endl << "pbtrf = ";
    //     // printMatrix(AB);
    // }

    // std::cout << std::endl << "mult = ";
    // printMatrix(blAH2);

    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = 0; i < n; i++){
    //         blAH2(i, j) = blAH2(i,j) - A(i, j);
    //     }
    // }

    // real_t normPbtrf = lange(tlapack::FROB_NORM, blAH2);

    // std::cout << "\nnorm of subtraction is " << normPbtrf/normPotrf << std::endl;


}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << std::endl
              << "example_pbtrf executed" << std::endl
              << std::endl;

    using std::size_t;
    int m, n;
    // printf("run< complex<double> >( %d, %d )", m, n);
    // run<std::complex<double>>(m, n);
    // printf("-----------------------\n");
    // Default arguments
    m = 9;
    n = m;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )", m, n);
    run<std::complex<float>>(m, n);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )", m, n);
    run<std::complex<double>>(m, n);
    printf("-----------------------\n");

    return 0;
}
