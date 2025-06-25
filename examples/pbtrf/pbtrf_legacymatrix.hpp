/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite band matrix AB.
/// @author Ella Addison-Taylor, Kyle Cunningham, Henricus Bouwmeester, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_PBTRF_LEGACYMATRIX_HH
#define TLAPACK_PBTRF_LEGACYMATRIX_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf0.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/potf2.hpp"


namespace tlapack {
/// @brief Options struct for pbtrf()
struct BlockedBandedCholeskyOpts : public EcOpts {
    constexpr BlockedBandedCholeskyOpts(const EcOpts& opts = {})
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
 *     *    *   a13  a24  a35  a46      *    *   u13  u24  u35  u46
 *     *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
 *    a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
 *
 * Similarly, if UPLO = 'L' the format of A is as follows:
 *
 *    On entry:                        On exit:
 *
 *    a11  a22  a33  a44  a55  a66     l11  l22  l33  l44  l55  l66
 *    a21  a32  a43  a54  a65   *      l21  l32  l43  l54  l65   *
 *    a31  a42  a53  a64   *    *      l31  l42  l53  l64   *    *
 *
 * Array elements marked * are not used by the routine.
 *
 * @ingroup variant_interface
 */

template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
int pbtrf_legacymatrix(uplo_t uplo,
           matrix_t& AB,
           const BlockedBandedCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    idx_t n = ncols(AB);
    idx_t kd = nrows(AB) - 1;
    const idx_t nb = opts.nb;

    if (nb < 1 || nb > kd) {
        return pbtf0(uplo, AB);
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

                auto AB00 =
                    slice(AB, range(kd - ib + 1, kd + 1), range(i, i + ib));

                AB00.ptr = &AB00.ptr[ib - 1];
                AB00.ldim -= 1;

                potf2(tlapack::Uplo::Upper, AB00);

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

                        auto AB01 = slice(
                            AB, range(kd - ib, kd),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));
                        AB01.ldim -= 1;

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, AB01);

                        auto AB11 = slice(
                            AB, range(kd + 1 - i2, kd + 1),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));

                        AB11.ptr = &AB11.ptr[i2 - 1];
                        AB11.ldim -= 1;

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), AB01, real_t(1), AB11);

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
                                work02(ii, jj) = AB(ii - jj, jj + i + kd);
                            }
                        }

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, work02);

                        auto AB12 =
                            slice(AB, range(kd - i2, kd),
                                  range(i + kd,
                                        std::min(static_cast<int>(i + kd + i3),
                                                 static_cast<int>(n))));
                        AB12.ldim -= 1;

                        auto AB01 = slice(
                            AB, range(kd - ib, kd),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));
                        AB01.ldim -= 1;

                        gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                             real_t(-1), AB01, work02, real_t(1), AB12);

                        auto AB22 = slice(
                            AB, range(kd - i3 + 1, kd + 1),
                            range(i + ib + i2,
                                  std::min(static_cast<int>(i + 2 * ib + i2),
                                           static_cast<int>(n))));
                        AB22.ptr = &AB22.ptr[i3 - 1];
                        AB22.ldim -= 1;

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), work02, real_t(1), AB22);

                        for (idx_t jj = 0; jj < i3; ++jj) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                AB(ii - jj, jj + i + kd) = work02(ii, jj);
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

                auto AB00 = slice(AB, range(0, ib), range(i, ib + i));
                AB00.ldim -= 1;

                potf2(tlapack::Uplo::Lower, AB00);

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
                        auto AB10 =
                            slice(AB, range(ib, ib + i2), range(i, ib + i));
                        AB10.ldim -= 1;

                        trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, AB10);

                        auto AB11 =
                            slice(AB, range(0, i2), range(i + ib, i + ib + i2));
                        AB11.ldim -= 1;

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), AB10,
                             real_t(1), AB11);
                    }

                    if (i3 > 0) {
                        auto AB10 =
                            slice(AB, range(ib, ib + i2), range(i, ib + i));
                        AB10.ldim -= 1;

                        auto AB20 = slice(AB, range(kd - i3 + 1, kd + 1),
                                          range(i, ib + i));
                        AB20.ptr = &AB20.ptr[i3 - 1];
                        AB20.ldim -= 1;

                        auto work20 = slice(work, range(0, i3), range(0, ib));

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < std::min(static_cast<int>(jj + 1),
                                               static_cast<int>(i3));
                                 ++ii) {
                                work20(ii, jj) = AB(kd - jj + ii, jj + i);
                            }
                        }

                        trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                             tlapack::Diag::NonUnit, real_t(1), AB00, work20);

                        auto AB21 = slice(AB, range(i2, i2 + i3),
                                          range(i + ib, i + ib + i2));
                        AB21.ldim -= 1;

                        gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                             real_t(-1), work20, AB10, real_t(1), AB21);

                        auto AB22 =
                            slice(AB, range(0, i3),
                                  range(i + ib + i2,
                                        i + ib + i2 + std::min((ib), (i3))));
                        AB22.ldim -= 1;

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                             real_t(1), AB22);

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < std::min(static_cast<int>(jj + 1),
                                               static_cast<int>(i3));
                                 ++ii) {
                                AB(kd - jj + ii, jj + i) = work20(ii, jj);
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

#endif // TLAPACK_PBTRF_LEGACYMATRIX_HH