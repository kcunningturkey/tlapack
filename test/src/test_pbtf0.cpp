/// @file test_pbtf0.cpp Test the Cholesky factorization of a symmetric positive
/// definite band matrix
/// @author Ella Addison-Taylor, Kyle Cunningham, University of Colorado Denver,
/// USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lanhe.hpp>

// Other routines
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/pbtf0.hpp>

using namespace tlapack;

TEMPLATE_TEST_CASE(
    "Cholesky factorization of a Hermitian positive-definite band matrix",
    "[pbtf0]",
    TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    using real_t = real_type<T>;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    const idx_t n = GENERATE(1, 3, 10, 40, 130);
    const idx_t kd = GENERATE(0, 1, 9, 10, 20, 31);
    const Uplo uplo = GENERATE(Uplo::Lower, Uplo::Upper);

    DYNAMIC_SECTION("n = " << n << " kd = " << kd << " uplo = " << uplo)
    {
        if (kd < n) {
            // eps is the machine precision, and tol is the tolerance we accept
            // for tests to pass
            const real_t eps = ulp<real_t>();
            const real_t tol = real_t(n) * eps;
            // const real_t tol = real_t(10) * real_t(n) * eps;

            // Create matrices
            std::vector<T> A_;
            auto A = new_matrix(A_, n, n);
            std::vector<T> A_copy_;
            auto A_copy = new_matrix(A_copy_, n, n);
            std::vector<T> AB_;
            auto AB = new_matrix(AB_, kd + 1, n);

            // Update A with random numbers, and make it positive definite
            mm.random(uplo, A);
            for (idx_t j = 0; j < n; ++j)
                A(j, j) += real_t(n) * real_t(n);

            // Filling matrix
            if (uplo == Uplo::Upper) {
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = j + 1; i < n; ++i) {
                        if constexpr (is_complex<T>) {
                            A(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                                        static_cast<real_t>(0xDEADBEEF));
                        }
                        else
                            A(i, j) = static_cast<T>(0xDEADBEEF);
                    }
                }
                for (idx_t j = kd; j < n; j++) {
                    for (idx_t i = 0; i < j - kd; i++) {
                        A(i, j) = static_cast<float>(0);
                    }
                }
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t i = std::max(
                             0, static_cast<int>(j) - static_cast<int>(kd));
                         i < j + 1; i++) {
                        AB(i + kd - j, j) = A(i, j);
                    }
                }
            }
            else {  // Uplo == Lower
                for (idx_t j = 0; j < n; ++j) {
                    for (idx_t i = 0; i < j; ++i) {
                        if constexpr (is_complex<T>) {
                            A(i, j) = T(static_cast<real_t>(0xDEADBEEF),
                                        static_cast<real_t>(0xDEADBEEF));
                        }
                        else
                            A(i, j) = static_cast<T>(0xDEADBEEF);
                    }
                }
                for (idx_t j = 0; j < n - kd; j++) {
                    for (idx_t i = kd + j + 1; i < n; i++) {
                        A(i, j) = static_cast<float>(0);
                    }
                }
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t i = j;
                         i < std::min(static_cast<int>(n),
                                      static_cast<int>(j + kd + 1));
                         i++) {
                        AB(i - j, j) = A(i, j);
                    }
                }
            }

            // Compute the norm of A
            lacpy(GENERAL, A, A_copy);
            real_t normA = lanhe(tlapack::FROB_NORM, uplo, A);

            // Run the Cholesky factorization
            int info = pbtf0(uplo, AB);

            // Check that the factorization was successful
            REQUIRE(info == 0);

            // put AB into A_copy
            if (uplo == tlapack::Uplo::Upper) {
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t i = std::max(
                             0, static_cast<int>(j) - static_cast<int>(kd));
                         i < j + 1; i++) {
                        A_copy(i, j) = AB(i + kd - j, j);
                    }
                }
                mult_uhu(A_copy);
            }
            else {
                for (idx_t j = 0; j < n; j++) {
                    for (idx_t i = j;
                         i < std::min(static_cast<int>(n),
                                      static_cast<int>(j + kd + 1));
                         i++) {
                        A_copy(i, j) = AB(i - j, j);
                    }
                }
                mult_llh(A_copy);
            }

            // Check that the factorization is correct
            for (idx_t i = 0; i < n; i++)
                for (idx_t j = 0; j < n; j++) {
                    if (uplo == Uplo::Lower && i >= j)
                        A_copy(i, j) -= A(i, j);
                    else if (uplo == Uplo::Upper && i <= j)
                        A_copy(i, j) -= A(i, j);
                }

            // Check for relative error: norm(A-cholesky(A))/norm(A)
            real_t error =
                tlapack::lanhe(tlapack::MAX_NORM, uplo, A_copy) / normA;
            CHECK(error <= tol);
        }
    }
}