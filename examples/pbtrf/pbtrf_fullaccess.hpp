/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite band matrix AB.
/// @author Ella Addison-Taylor, Kyle Cunningham, Henricus Bouwmeester, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_PBTRF_FULLACCESS_HH
#define TLAPACK_PBTRF_FULLACCESS_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf0.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/potf2.hpp"
#include "pbtf0_fullaccess.hpp"


namespace tlapack {

template <typename matrix_t>
void printMatrix3(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
    std::cout << std::endl;
}
/// @brief Options struct for pbtrf()
struct BlockedBandedTestCholeskyOpts : public EcOpts {
    constexpr BlockedBandedTestCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts) {};

    size_t nb = 32;  ///< Block size
};

/** Computes the Cholesky factorization of a Hermitian
 * positive definite band matrix A.
 *
 * The factorization has the form
 *      $A = U^H U,$ if uplo = Upper, or
 *      $A = L L^H,$ if uplo = Lower,
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in] uplo
 *      - Uplo::Upper: Upper triangle of A is referenced;
 *      - Uplo::Lower: Lower triangle of A is referenced.
 *
 * @param[in,out] AB
 *       AB is an array, dimension (kd+1, N)
 *      On entry, the Hermitian positive definite band matrix AB of size kd+1-by-n.
 *
 *      - If uplo = Uplo::Upper, AB(i + kd - j, j) = A(i, j) for max(0,j-kd)<=i<=j
 *
 *      - If uplo = Uplo::Lower, AB(i - j, j) = A(i, j) for j<=i<=min(n,j+kd+1)
 *
 *      - On successful exit, the factor U or L from the Cholesky
 *      factorization $A = U^H U$ or $A = L L^H.$
 *
 * @param[in] opts Options.
 *      Define the behavior of nb for pbtrf_legacymatrix.     
 *
 * @return 0: successful exit.
 * 
 * @return i, 0 < i <= n, if the leading minor of order i is not
 *      positive definite, and the factorization could not be completed.
 * 
 * @par Further Details
 * 
 * The band storage scheme is illustrated by the following example, when
 * N = 6, KD = 2, and UPLO = 'U':
 *
 *    On entry:                        On exit:
 *
 *     *    *   a02  a13  a24  a35      *    *   u02  u13  u24  u35
 *     *   a01  a12  a23  a34  a45      *   u01  u12  u23  u34  u45
 *    a00  a11  a22  a33  a44  a55     u00  u11  u22  u33  u44  u55
 *
 * Similarly, if UPLO = 'L' the format of A is as follows:
 *
 *    On entry:                        On exit:
 *
 *    a00  a11  a22  a33  a44  a55     l00  l11  l22  l33  l44  l55
 *    a10  a21  a32  a43  a54   *      l10  l21  l32  l43  l54   *
 *    a20  a31  a42  a53   *    *      l20  l31  l42  l53   *    *
 *
 * Array elements marked * are not used by the routine.
 *
 * @ingroup variant_interface
 */

template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
int pbtrf_fullaccess(uplo_t uplo,
           matrix_t& A,
           std::size_t& kd,
           const BlockedBandedTestCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    idx_t n = ncols(A);
    const idx_t nb = opts.nb;

    if (nb < 1 || nb > kd) {
        return pbtf0_fullaccess(uplo, A, kd);
        
    }
    else {
        std::vector<T> work_(nb * nb);
        for (idx_t ii = 0; ii < nb * nb; ++ii) {
            if constexpr (tlapack::is_complex<T>) {
                work_[ii] = T(0, 0);
            }
            else
                work_[ii] = 0;
        }
        auto work = new_matrix(work_, nb, nb);

        if (uplo == tlapack::Uplo::Upper) {
            for (idx_t i = 0; i < n; i += nb) {
                // ib = min(nb, n - i)
                idx_t ib;
                if (n < nb + i) {
                    ib = n - i;
                }
                else  
                {
                    ib = nb;
                }

                auto A00 =
                    slice(A, range(i, ib + i), range(i, min(i + ib, n)));
                    std::cout << "A00 = " << std::endl;
                    printMatrix3(A00);
                    std::cout << std::endl;


                potf2(tlapack::Uplo::Upper, A00);
                std::cout << "done potf2" << std::endl;
                std::cout << "i + ib = " << i + ib << " and n = " << n << std::endl;
                if (i + ib < n) {
                    
                    // i2 = min(kd-ib, n-i-ib)
                    idx_t i2;
                    if (kd + i < n) {
                        i2 = kd - ib;
                    }
                    else {
                        i2 = n - i - ib;
                    }

                    if (i2 > 0) {

                        std::cout << "A01 = " << std::endl;
                        auto A01 = slice(
                            A, range(i, ib + i),
                            range(i + ib, min(i + ib + i2,
                                                   n)));
                        printMatrix3(A01);

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A01);

                        std::cout << "A11 = " << std::endl;
                        auto A11 = slice(
                            A, range(i + ib, i + kd),
                            range(i + ib, min(i + kd,
                                                   n)));
                        printMatrix3(A11);

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), A01, real_t(1), A11);

                    }
                    // i3 = min(ib, n-i-kd)
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

                    if (i3 > 0) {
                        
                        auto work02 = slice(work, range(0, ib), range(0, i3));

                        for (idx_t jj = 0; jj < i3; jj++) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                std::cout << "work02(" << ii << ", " << jj << ") = A(" << ii - jj << ", " << jj + i + kd << ")" << std::endl;
                                // work02(ii, jj) = A(ii - jj, jj + i + kd);
                                work02(ii, jj) = A(i+ii, i+kd+jj);
                            }
                        }

                        std::cout << "work = " << std::endl;
                        printMatrix3(work02);
                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, work02);

                        std::cout << "A12 = " << std::endl;
                        auto A12 =
                            slice(A, range(i + ib, i + kd),
                                  range(i + kd,
                                        min(i + kd + i3,
                                                 n)));
                        printMatrix3(A12);

                        auto A01 = slice(
                            A, range(i, ib + i),
                            range(i + ib, min(i + ib + i2,
                                                   n)));

                        gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                             real_t(-1), A01, work02, real_t(1), A12);

                             std::cout << "A22 = " << std::endl;
                        auto A22 = slice(
                            A, range(i + kd,
                                        min(i + kd + i3,
                                                 n)),
                            range(i + kd,
                                        min(i + kd + i3,
                                                 n)));

                        printMatrix3(A22);

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), work02, real_t(1), A22);

                        for (idx_t jj = 0; jj < i3; ++jj) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                A(i+ii, i+kd+jj) = work02(ii, jj);
                            }
                        }
                    }
                }
            }
        }
        else { // uplo == Lower
            for (idx_t i = 0; i < n; i += nb)
            {
                idx_t ib;
                if (nb + i < n) {
                    ib = nb;
                }
                else {
                    ib = n - i;
                }

                std::cout << "A00 = rows (" << i << ", " << ib << ") cols = (" << i << ", " << min(ib + i, n) << std::endl;
                auto A00 = slice(A, range(i, i + ib), range(i, min(ib + i, n)));
                printMatrix3(A00);

                potf2(tlapack::Uplo::Lower, A00);

                if (i + ib <= n) {
                    idx_t i2;
                    if (kd + i < n) {
                        i2 = kd - ib;
                    }
                    else {
                        i2 = n - i - ib;
                    }

                    idx_t i3;
                    if (ib + i + kd < n) {
                        i3 = ib;
                    }
                    else if (n < i + kd) {
                        i3 = 0;
                    }
                    else {
                        i3 = n - i - kd;
                    }

                    if (i2 > 0) {
                        std::cout << "A10 = " << std::endl;
                        auto A10 =
                            slice(A, range(ib + i, ib + i2 + i), range(i, ib + i));
                        printMatrix3(A10);

                        trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A10);

                            std::cout << "A11 = " << std::endl;

                        auto A11 =
                            slice(A, range(ib + i, ib + i2 + i), range(i + ib, i + ib + i2));
                            printMatrix3(A11);

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                             real_t(1), A11);
                    }

                    if (i3 > 0) {
                        auto A10 =
                            slice(A, range(ib + i, ib + i2 + i), range(i, ib + i));

                            std::cout << "work20 = " << std::endl;

                        auto work20 = slice(work, range(0, i3), range(0, ib));
                        

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < min(jj + 1,
                                               i3);
                                 ++ii) {
                                    std::cout << "work20(" << ii << ", " << jj << ") = A(" << i + kd + ii << ", " << jj + i << ") " << std::endl;
                                work20(ii, jj) = A(i + kd + ii, jj + i);
                            }
                        }
printMatrix3(work20);
                        trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                             tlapack::Diag::NonUnit, real_t(1), A00, work20);

                             std::cout << "A21 = " << std::endl;
                        auto A21 = slice(A, range(kd + i, kd + i + i3),
                                          range(i + ib, i + ib + i2));
                        printMatrix3(A21);

                        gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                             real_t(-1), work20, A10, real_t(1), A21);

                        std::cout << "A22 = " << std::endl;
                        auto A22 =
                            slice(A, range(kd + i, kd + i + i3),
                                  range(kd + i, kd + i + i3));
                        printMatrix3(A22);

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                             real_t(1), A22);

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < min(jj + 1,
                                               i3);
                                 ++ii) {
                                A(i + kd + ii, jj + i) = work20(ii, jj);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
}  // namespace tlapack

#endif // TLAPACK_PBTRF_FULLACCESS_HH