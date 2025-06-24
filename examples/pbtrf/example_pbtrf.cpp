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
#include <tlapack/lapack/ldam1.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/blas/gemm.hpp>

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

    // std::size_t kd = 7;
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

    for (idx_t j = 0; j < kd + 1; ++j) {
        for ( idx_t i = 0; i < n; ++i) {
            if constexpr (tlapack::is_complex<T>) {
                AB(j, i) = T(static_cast<real_t>(0xDEADBEEF), static_cast<real_t>(0xDEADBEEF));
            }
            else {
                AB(j, i) = static_cast<real_t>(0xDEADBEEF);
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

        for (idx_t j = kd; j < n; j++) {
        for (idx_t i = 0; i < j-kd; i++) {
            A(i, j) = static_cast<float>(0);
        }
    }


    // std::cout << std::endl << "AB before = ";
    // printMatrix(AB);  

     potf2(tlapack::Uplo::Upper, A);
    std::cout << std::endl << "potrf A before = ";
    printMatrix(A);

    // real_t normA = lange(tlapack::FROB_NORM, A);
    // lacpy(tlapack::Uplo::General, AB, blAH);

    // lacpy(tlapack::Uplo::General, A, blAH2);
    
    // std::cout << "blAH" << std::endl;
    // printMatrix(blAH);



    // pbtrf(uplo, AB, kd);

    // pbtf2(uplo, blAH);

    // std::cout << "\npbtrf" << std::endl;
    // printMatrix(A);

    // std::cout << "\nlevel 0 factor" << std::endl;
    // printMatrix(blAH);

    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         A(i, j) = AB(i + kd - j, j);
    //     }
    // }

    // mult_uhu(A);

    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = 0; i < n; i++){
    //         blAH2(i, j) = blAH2(i,j) - A(i, j);
    //     }
    // } 
    
    // std::cout << "blAH2" << std::endl;
    // printMatrix(blAH2);

    // real_t normB = lange(tlapack::FROB_NORM, blAH2);

    // std::cout << "\nThe norm is " << normB/normA << std::endl;

    std::cout << "\nPrint AB before (:" << std::endl;
    printMatrix(AB);

    // std::cout << "\nPrint A before (:" << std::endl;
    // printMatrix(A);
    // std::cout << std::endl;
    // for(idx_t i = 0; i < n; i += nb)
    // {
    //     //idx_t ib = std::min(static_cast<int>(nb), static_cast<int>(n - i));
    //     if (n < i) 
    //     {
    //         idx_t ib = 0;
    //     }
    //     else if (nb+i < n)
    //     {
    //         idx_t ib = nb;
    //     }

    //     std::cout << "ib = " << ib << std::endl;
    // }
    
    for(idx_t i = 0; i < n; i += nb)
    // for(idx_t i = 0; i < 1; i += nb)
    {
        std::cout << "this is loop num = " << i << std::endl;
        // idx_t ib = std::min(static_cast<int>(nb), static_cast<int>(n - i));
        idx_t ib;
        if (n < nb + i) 
        {
          ib = n - i;
        }
        else //(nb + i < n)
        {
           ib = nb;
        }

        std::cout << "\n\n\n\nib = " << ib << std::endl;
        std::cout << "size of AB00 is " << kd - ib + 1 << ", " << kd + 1 << " x " << i << ", " << i + ib << std::endl;

        auto AB00 = slice(AB, range(kd - ib +1, kd + 1), range(i, i + ib));
        std::cout << "AB00 = " << std::endl;
        printMatrix(AB00);
        std::cout << std::endl;

        AB00.ptr = &AB00.ptr[ib - 1];
        AB00.ldim -= 1;
        std::cout << "starting potf2" << std::endl;
        std::cout << "AB00 ptr = " << AB00.ptr[0] << std::endl;

        potf2(tlapack::Uplo::Upper, AB00);
        std::cout << "done potf2" << std::endl;



        // idx_t i2 = std::min(static_cast<int>(kd-ib), static_cast<int>(n - i - ib));
        idx_t i2;
        if (kd + i < n) {
            i2 = kd - ib;
        }
        else {
            i2 = n - i - ib;
        }
        std::cout << "i2 = " << i2 << std::endl;
        if (i2 > 0 ){

        // auto AB01 = slice(AB, range( kd -i2 -1 , kd ), range(i + ib, i + kd));
        auto AB01 = slice(AB, range( kd -ib , kd ), range(i + ib, i + kd));
        AB01.ldim -= 1;

        std::cout << "AB01 = " << std::endl;
        printMatrix(AB01);

        trsm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::ConjTrans, 
             tlapack::Diag::NonUnit, real_t(1), AB00, AB01);

        std::cout << "\nAB01 after = " << std::endl;
        printMatrix(AB01);
        std::cout << std::endl;

        // //CHECKING TRSM
        // trmm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::ConjTrans, 
        //      tlapack::Diag::NonUnit, real_t(1), AB00, AB01);

        // std::cout << "AB01 restored = " << std::endl;
        // printMatrix(AB01);
        // std::cout << std::endl;

        // std::cout << "AB restored = " << std::endl;
        // printMatrix(AB);
        // std::cout << std::endl;

        auto AB11 = slice(AB, range(kd+1-i2 , kd + 1 ), range(i + ib, i + kd));
        AB11.ptr = &AB11.ptr[i2 - 1];
        AB11.ldim -= 1;

                std::cout << "AB11 = " << std::endl;
        printMatrix(AB11);

        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans, 
            real_t(-1), AB01, real_t(1), AB11);

        std::cout << "AB11 = " << std::endl;
        printMatrix(AB11);
        std::cout << std::endl;  

        }

       // int i3 = std::min(static_cast<int>(ib), static_cast<int>(n- i - kd)); // change to int i hate unsigned ints omg
       idx_t i3;
       if (ib + i + kd < n) {
            i3 = ib;
       }
       else if (n > kd + i) {
            i3 = n - i - kd;
       }
       else {
            i3 = 0;
       }
        std::cout << "ib = " << ib << std::endl;
        std::cout << "n - i - kd = " << n << " - " << i << " - " << kd << " = " << static_cast<int>(n) - static_cast<int>(i) - static_cast<int>(kd) << std::endl;
        std::cout << "i3 = " << i3 << std::endl;
        std::cout << "test = " << 0 - 5 << std::endl;
        if (i3 > 0) {

            auto AB02 = slice(AB, range(0, kd - i2), range(i + kd, i + kd + i3));
            AB02.ldim -= 1;

            std::vector<T> work_;
            auto work = new_matrix(work_, kd - i2, i3);

            for (idx_t i = 0; i < kd - i2; ++i) {
                for (idx_t j = 0; j < i3; ++j) {
                    work(i, j) = 0;
                }
            }

            for(idx_t jj = 0; jj < i3; jj++){
                for(idx_t ii = jj; ii < ib; ++ii){
                    work(ii, jj) = AB(ii-jj, jj+i+kd);
                }
            }

            std::cout << "AB02 before = " << std::endl;
            printMatrix(AB02);
            std::cout << std::endl;

            std::cout << "work before = " << std::endl;
            printMatrix(work);
            std::cout << std::endl;

            trsm(tlapack::Side::Left, tlapack::Uplo::Upper, tlapack::Op::ConjTrans, tlapack::Diag::NonUnit, real_t(1), AB00, work);

            std::cout << "work after = " << std::endl;
            printMatrix(work);
            std::cout << std::endl;

            auto AB12 = slice(AB, range(kd-i2 , kd ), range(i + kd, i + kd + i3));
            AB12.ldim -= 1;

            std::cout << "AB12 = " << std::endl;
            printMatrix(AB12);
            std::cout << std::endl;

            auto AB01 = slice(AB, range( kd -ib , kd ), range(i + ib, i + kd));
            AB01.ldim -= 1;

            gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans, real_t(-1), AB01, work, real_t(1), AB12);

            std::cout << "AB12 after = " << std::endl;
            printMatrix(AB12);
            std::cout << std::endl;

            std::cout << "about to slice ab22" << std::endl;
            std::cout << "slicing rows = " << kd - i3 + 1 << " - " << kd + 1 << " slicing columns = " << i + ib + i2 << " - " << std::min(static_cast<int>(i + 2 *ib + i2), static_cast<int>(n)) << std::endl;
            auto AB22 = slice(AB, range(kd- i3 + 1 , kd + 1 ), range(i + ib + i2, std::min(static_cast<int>(i + 2 *ib + i2), static_cast<int>(n))));
            std::cout << "done slice" << std::endl;
            AB22.ptr = &AB22.ptr[i3 - 1];
            AB22.ldim -= 1;

            std::cout << "AB22 = " << std::endl;
            printMatrix(AB22);
            std::cout << std::endl;

            herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans, real_t(-1), work, real_t(1), AB22);

            for (idx_t jj = 0; jj < i3; ++jj) {
                for (idx_t ii = jj; ii < ib; ++ii) {
                    AB(ii-jj, jj+i+kd) = work(ii, jj);
                }
            }

            std::cout << "AB with work = " << std::endl;
            printMatrix(AB);
            std::cout << std::endl;

        }

    }




    std::cout << "\nPrint AB after" << std::endl;
    printMatrix(AB);

    std::cout << "\nPrint A after " << std::endl;
    printMatrix(A);

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

    using idx_t = size_t;

    idx_t m, n, kd, nb;
    // printf("run< complex<double> >( %d, %d )", m, n);
    // run<std::complex<double>>(m, n);
    // printf("-----------------------\n");
    // Default arguments
    m = 13;
    n = m;
    kd = 7;
    nb = 5;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    run<float>(m, n, kd, nb);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    // run<double>(m, n, kd);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    // run<long double>(m, n, kd);
    // printf("-----------------------\n");

    // printf("run< complex<float> >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    // run<std::complex<float>>(m, n, kd);
    // printf("-----------------------\n");

    // printf("run< complex<double> >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    // run<std::complex<double>>(m, n, kd);
    // printf("-----------------------\n");

    return 0;
}
