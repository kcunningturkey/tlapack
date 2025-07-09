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
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/trsm_tri.hpp>
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

//------------------------------------------------------------------------------
template <typename T, typename uplo_t>
void run(Side sideA, uplo_t uplo, Op transA, Diag diagA, size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    tlapack::Create<matrix_t> new_matrix;

    // Create A
    std::vector<T> A_;
    auto A = new_matrix(A_, n, n);

    // Create B
    std::vector<T> B_;
    auto B = new_matrix(B_, n, n);

    // Fill A with DEADBEEF
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if constexpr (tlapack::is_complex<T>) {
                // A(i, j) = T(0, 0);
                A(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                            static_cast<real_t>(0xDEADBEEF));
            }
            else {
                // A(i, j) = T(0);
                A(i, j) = T(static_cast<real_t>(0xDEADBEEF));
            }
        }
    }

    // Fill B with DEADBEEF
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if constexpr (tlapack::is_complex<T>) {
                // B(i, j) = T(0, 0);
                B(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                            static_cast<real_t>(0xDEADBEEF));
            }
            else {
                // B(i, j) = T(0);
                B(i, j) = T(static_cast<real_t>(0xDEADBEEF));
            }
        }
    }

    if (transA ==  //---------------------------------NoTrans------------------------------------
        tlapack::Op::NoTrans) {
        if (uplo ==  //-------------------------------Lower-------------------------------------
            tlapack::Uplo::Lower) {
            // Fill the lower part of A
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

            // Fill the lower part of B
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
            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(GENERAL, B, X);

            std::vector<T> A_0_;
            auto A_0 = new_matrix(A_0_, n, n);
            lacpy(GENERAL, A, A_0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        A_0(i, j) = T(0, 0);
                        B(i, j) = T(0, 0);
                    }
                    else {
                        A_0(i, j) = T(0);
                        B(i, j) = T(0);
                    }
            real_t normA = lange(Norm::Fro, A_0);
            real_t normB = lange(Norm::Fro, B);

            trsm_tri(sideA, uplo, transA, diagA, A, X);

            // Remove 0xDEADBEEF
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF),
                                     static_cast<real_t>(0xDEADBEEF));
                    }
                    else {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF));
                    }

            // Check
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;
        }
        else {  //-----------------------------------------------------------Upper-------------------------------------
            // Fill the Upper part of A
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            // Fill the upper part of B
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(GENERAL, B, X);

            std::vector<T> A_0_;
            auto A_0 = new_matrix(A_0_, n, n);
            lacpy(GENERAL, A, A_0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j + 1; i < n; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        A_0(i, j) = T(0, 0);
                        B(i, j) = T(0, 0);
                    }
                    else {
                        A_0(i, j) = T(0);
                        B(i, j) = T(0);
                    }
            real_t normA = lange(Norm::Fro, A_0);
            real_t normB = lange(Norm::Fro, B);

            trsm_tri(sideA, uplo, transA, diagA, A, X);

            // Remove 0xDEADBEEF
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j+1; i < n; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF),
                                     static_cast<real_t>(0xDEADBEEF));
                    }
                    else {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF));
                    }

            // Check
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;  // HENC: Is this correct?
        }
    }
    else if ( 
        transA == tlapack::Op::Trans) //-------------------------------Trans-------------------------------------
    {
        if (uplo == tlapack::Uplo::Lower) { //-----------------------------------Lower-------------------------------------
            // Fill the Upper part of A
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            // Fill the lower part of B
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

            std::cout << "A = " << std::endl;
            printMatrix(A);
            std::cout << "B = " << std::endl;
            printMatrix(B);

            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(GENERAL, B, X);

            std::cout << "X = " << std::endl;
            printMatrix(X);

            std::vector<T> A_0_;
            auto A_0 = new_matrix(A_0_, n, n);
            lacpy(GENERAL, A, A_0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = j+1; i < n; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        A_0(i, j) = T(0, 0);
                        B(j, i) = T(0, 0);
                    }
                    else {
                        A_0(i, j) = T(0);
                        B(j, i) = T(0);
                    }
            std::cout << "A_0 zeros = " << std::endl;
            printMatrix(A_0);

            std::cout << "B zeros = " << std::endl;
            printMatrix(B);
            
            
            real_t normA = lange(Norm::Fro, A_0);
            real_t normB = lange(Norm::Fro, B);

            std::cout << "X before trsm_tri  = " << std::endl;
            printMatrix(X);

            trsm_tri(sideA, uplo, transA, diagA, A, X);

            std::cout << "X after trsm_tri  = " << std::endl;
            printMatrix(X);        
            // Remove 0xDEADBEEF

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF),
                                     static_cast<real_t>(0xDEADBEEF));
                    }
                    else {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF));
                    }

            std::cout << "X no DEADBEEF  = " << std::endl;
            printMatrix(X); 

            // Check
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;  
        }
        else {  //--------------------------------------------------------------Upper-------------------------------------
            // Fill the Lower part of A
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

            // Fill the upper part of B
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(uplo, B, X);

            std::vector<T> A_0_;
            auto A_0 = new_matrix(A_0_, n, n);
            lacpy(GENERAL, A, A_0);

            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        A_0(i, j) = T(0, 0);
                        B(i, j) = T(0, 0);
                    }
                    else {
                        A_0(i, j) = T(0);
                        B(i, j) = T(0);
                    }

            real_t normA = lange(Norm::Fro, A_0);
            real_t normB = lange(Norm::Fro, B);

            trsm_tri(sideA, uplo, transA, diagA, A, X);

            // Remove 0xDEADBEEF
            for (idx_t j = 0; j < n; ++j)
                for (idx_t i = 0; i < j; ++i)
                    if constexpr (tlapack::is_complex<T>) {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF),
                                     static_cast<real_t>(0xDEADBEEF));
                    }
                    else {
                        X(i, j) -= T(static_cast<real_t>(0xDEADBEEF));
                    }

            // Check
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;  // HENC: Is this correct?
        }
    }
    else  //--------------------------------------------------------------ConjTrans-----------------------------------
    {
        if (uplo ==
            tlapack::Uplo::
                Lower) {  //-------------------------------Lower-------------------------------------
            // Fill the Upper part of A
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            // Fill the lower part of B
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

            // Ensure the B is in the column space of A
            // trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit,
            // real_t(1),
            //      A, B);

            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(uplo, B, X);

            real_t normA = lange(Norm::Fro, A);
            real_t normB = lange(Norm::Fro, B);

            // printf("A = ");
            // printMatrix(A);

            // printf("X = ");
            // printMatrix(X);

            // printf("B = ");
            // printMatrix(B);

            // trsm_tri(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit, A,
            // X);
            std::cout << sideA << ", " << uplo << ", " << transA << std::endl;
            trsm_tri(sideA, uplo, transA, diagA, A, X);

            // Check
            // trmm(Side::Left, Uplo::Lower, Op::NoTrans, Diag::NonUnit,
            // real_t(1),
            //      A, X);
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            // printf("X = ");
            // printMatrix(X);

            // printf("B = ");
            // printMatrix(B);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            // std::cout << "Norm = " << normB / normB_orig << std::endl;
            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;  // HENC: Is this correct?
        }

        else {  //-----------------------------------------------------------Upper-------------------------------------

            // Fill the Lower part of A
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

            // Fill the upper part of B
            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = i; j < n; j++) {
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

            std::vector<T> X_;
            auto X = new_matrix(X_, n, n);
            lacpy(uplo, B, X);

            real_t normA = lange(Norm::Fro, A);
            real_t normB = lange(Norm::Fro, B);

            // printf("A = ");
            // printMatrix(A);

            std::cout << sideA << ", " << uplo << ", " << transA << std::endl;

            trsm_tri(sideA, uplo, transA, diagA, A, X);

            // printf("X = ");
            // printMatrix(X);

            // printf("B = ");
            // printMatrix(B);

            // Check
            trmm(sideA, uplo, transA, diagA, real_t(1), A, X);

            for (idx_t i = 0; i < n; i++) {
                for (idx_t j = 0; j < n; j++) {
                    X(i, j) -= B(i, j);
                }
            }

            real_t normX = lange(Norm::Fro, X);

            std::cout << "Norm = " << normX / (normB + normA * normX)
                      << std::endl;  // HENC: Is this correct?
        }
    }
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    idx_t n = 13;
    tlapack::Uplo uplo;
    tlapack::Diag diagA;
    tlapack::Side sideA;
    tlapack::Op transA;

    transA = tlapack::Op::NoTrans;
    // transA = tlapack::Op::Trans;
    // transA = tlapack::Op::ConjTrans;

    sideA = tlapack::Side::Left;
    // sideA = tlapack::Side::Right;

    diagA = tlapack::Diag::NonUnit;
    // diagA = tlapack::Diag::Unit;

    uplo = tlapack::Uplo::Upper;

    printf("--------Upper--------\n");
    printf("run< float  >( %d, %d )\n", static_cast<int>(n),
           static_cast<int>(n));
    run<float>(sideA, uplo, transA, diagA, n);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<double>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<long double>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< complex<float> >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<std::complex<float>>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< complex<double> >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<std::complex<double>>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    //-------------------------------------------------------------------Lower---------------------------------------------
    // // transA = tlapack::Op::NoTrans;
    // // transA = tlapack::Op::Trans;
    // transA = tlapack::Op::ConjTrans;

    // sideA = tlapack::Side::Left;
    // // sideA = tlapack::Side::Right;

    // diagA = tlapack::Diag::NonUnit;
    // // diagA = tlapack::Diag::Unit;
    std::cout << std::endl;
    uplo = tlapack::Uplo::Lower;

    printf("--------Lower--------\n");

    printf("run< float  >( %d, %d )\n", static_cast<int>(n),
           static_cast<int>(n));
    run<float>(sideA, uplo, transA, diagA, n);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<double>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<long double>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< complex<float> >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<std::complex<float>>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    // printf("run< complex<double> >( %d, %d )\n", static_cast<int>(n),
    //        static_cast<int>(n));
    // run<std::complex<double>>(sideA, uplo, transA, diagA, n);
    // printf("-----------------------\n");

    return 0;
}