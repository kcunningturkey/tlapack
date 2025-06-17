/// @file template.cpp
/// @author AUTHOR, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK

#include <tlapack/lapack/pbstf.hpp>

// C++ Includes
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
        for (idx_t j = 0; j < n; ++j){
            std::cout << A(i, j) << " ";
        }
    }
}

//------------------------------------------------------------------------------
// template <typename real_t>
template <typename T>
void run(size_t m, size_t n)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<real_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Matrix A
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);

    // Matrix AB
    idx_t kd = 3;
    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    // Banded upper triangular matrix
    for (idx_t j = 0; j < n; j++) {
        real_t real_diag;
        real_diag = j * j + 10;  // Strong Diagonal
        A(j, j) = real_diag;

        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j; i++) {
            A(i, j) = T(i + 5, j);
        }

        for (idx_t i = j + 1; i < m; i++) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF),
                        static_cast<float>(0xDEADBEEF));
        }
    }

    // Creating matrix AB from A
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            AB(i + kd - j, j) = A(i, j);
        }
    }

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);

        // std::cout << std::endl << "AB = ";
        // printMatrix(AB);
    }
    // Calling pbstf
    pbstf(tlapack::UPPER_TRIANGLE, AB);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;
    size_t m, n;

    // Default arguments

    n = 7;
    m = 7;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    // printf("run< float  >( %d, %d )", m, n);
    // run<float>(m, n);
    // printf("-----------------------\n");

    std::cout << "run< complex<float> >( " << n << " )" << std::endl;
    run<std::complex<float> >(m, n);
    std::cout << "-----------------------" << std::endl;

    // std::cout << "run >( " << n << " )" << std::endl;
    // run(m, n);
    // std::cout << "-----------------------" << std::endl;

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");

    return 0;
}
