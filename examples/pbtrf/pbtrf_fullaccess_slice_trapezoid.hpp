/// @file potrf.hpp Computes the Cholesky factorization of a Hermitian positive
/// definite band matrix AB.
/// @author Ella Addison-Taylor, Kyle Cunningham, Henricus Bouwmeester,
/// University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_PBTRF_FULLACCESS_SLICE_TRAPEZOID_HH
#define TLAPACK_PBTRF_FULLACCESS_SLICE_TRAPEZOID_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf0.hpp"
#include "tlapack/lapack/potf2.hpp"

namespace tlapack {
/// @brief Options struct for pbtrf()
struct BlockedBandedFullCholeskyOpts : public EcOpts {
    constexpr BlockedBandedFullCholeskyOpts(const EcOpts& opts = {})
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
 *      On entry, the Hermitian positive definite band matrix AB of size
 * kd+1-by-n.
 *
 *      - If uplo = Uplo::Upper, AB(i + kd - j, j) = A(i, j) for
 * max(0,j-kd)<=i<=j
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

template <typename matrix_t>
void printMatrix2(const matrix_t& A)
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

#define isSlice_2(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value

// Slice LegacyMatrix
template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice_2(SliceSpecRow) && isSlice_2(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice_trapezoid(LegacyMatrix<T, idx_t, layout>& A,
                               SliceSpecRow&& rows,
                               SliceSpecCol&& cols) noexcept
{
    idx_t ptr_offset = 0;

    idx_t ABcols_first = cols.first;
    idx_t ABcols_second = cols.second;

    idx_t ABrows_first;
    idx_t ABrows_second = cols.second;

    if (rows.first == cols.first) {
        std::cout << "diagonals" << std::endl;
        ptr_offset = rows.second - rows.first - 1;
        ABrows_first = nrows(A) - (rows.second - rows.first);
        ABrows_second = nrows(A);
    }
    else if (rows.first != cols.first && cols.first - 1 == rows.second) {
        std::cout << "green" << std::endl;
        ABrows_first = 0;
        ABrows_second = rows.second - rows.first;
    }
    else {
        // std::cout << "rectnagles" << std::endl;
        // ABrows_first = cols.second - cols.first;
        // ABrows_second =
        //     min((rows.second - rows.first) + ABrows_first, nrows(A));
        std::cout << "rectnagles" << std::endl;
        ABrows_first = cols.second - cols.first;
        ABrows_second =
            (rows.second - rows.first) + ABrows_first;
    }
    // else if (cols.first + 1 == nrows(A) && ) {
    //     std::cout << "green" << std::endl;
    //     ABrows_first = 0;
    //     ABrows_second = rows.second - rows.first;
    // }
    // else {
    //     std::cout << "rectnagles" << std::endl;
    //     ABrows_first = cols.second - cols.first;
    //     ABrows_second = min((rows.second - rows.first) + ABrows_first,
    //     nrows(A));

    // }

    std::cout << "range(" << ABrows_first << ", " << ABrows_second
              << ") and range(" << ABcols_first << ", " << ABcols_second << ")"
              << std::endl;
    return LegacyMatrix<T, idx_t, layout>(
        ABrows_second - ABrows_first, ABcols_second - ABcols_first,
        (layout == Layout::ColMajor)
            ? &A.ptr[ptr_offset + ABrows_first + ABcols_first * A.ldim]
            : &A.ptr[(ptr_offset + ABrows_first) * A.ldim + ABcols_first],
        A.ldim - 1);
}

#undef isSlice

template <TLAPACK_UPLO uplo_t, TLAPACK_MATRIX matrix_t>
int pbtrf_fullaccess_slice_trapezoid(
    uplo_t uplo, matrix_t& A, const BlockedBandedFullCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    idx_t n = ncols(A);
    idx_t kd = nrows(A) - 1;
    const idx_t nb = opts.nb;

    if (nb < 1 || nb > kd) {
        // std::cout << "HENC" << std::endl;
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
                else {
                    ib = nb;
                }
                std::cout << "AB = " << std::endl;
                printMatrix2(A);
                std::cout << "range(" << i << ", " << ib + i << "), range(" << i
                          << ", min(" << i + ib << ", " << n << ")"
                          << std::endl;
                auto A00 =
                    slice_trapezoid(A, range(i, ib + i), range(i, min(i + ib, n)));
                std::cout << "A00 = " << std::endl;
                printMatrix2(A00);

                potf2(tlapack::Uplo::Upper, A00);
            
                std::cout << "potf2(tlapack::Uplo::Upper, A00);" << std::endl;///////////

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
                        auto A01 = slice_trapezoid(A, range(i, ib + i),
                                                   range(i + ib, i + ib + i2));
                        std::cout << "A01 = " << std::endl;
                        printMatrix2(A01);

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A01);
                        std::cout << "trsm (A00, A01)" << std::endl;///////////////

                        auto A11 = slice_trapezoid(A, range(i + ib, i + kd),
                                                   range(i + ib, i + kd));
                        std::cout << "A11 = " << std::endl;
                        printMatrix2(A11);

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), A01, real_t(1), A11);
                        std::cout << "herk (A01, A11)" << std::endl;///////////////
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
                        std::cout << "i = " << i << " i + ib = " << i + ib
                                  << std::endl;
                        auto A02 = slice_trapezoid(A, range(i, i + ib),
                                                   range(i + kd, i + kd + i3));
                        std::cout << "A02 = " << std::endl;
                        printMatrix2(A02);
                        auto work02 = slice(work, range(0, ib), range(0, i3));
                        for (idx_t jj = 0; jj < i3; jj++)
                            for (idx_t ii = jj; ii < ib; ++ii)
                                work02(ii, jj) = A02(ii, jj);
                        // auto work02 = slice(work, range(0, ib), range(0,
                        // i3));

                        // for (idx_t jj = 0; jj < i3; jj++) {
                        //     for (idx_t ii = jj; ii < ib; ++ii) {
                        //         std::cout << "work(" << ii << ", " << jj <<
                        //         ") = A(" << i + ii << ", " << i + kd + jj <<
                        //         std::endl; work02(ii, jj) = A(i+ii, i+kd+jj);
                        //     }
                        // }
                        std::cout << "work = " << std::endl;
                        printMatrix2(work02);
                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, work02);
                        std::cout << "trsm (A00, work02)" << std::endl;////////////////////

                        auto A12 =
                            slice_trapezoid(A, range(i + ib, i + kd),
                                            range(i + kd, min(i + kd + i3, n)));

                        std::cout << "A12 = rows = (" << i + ib << ", " << i + kd << ") cols = (" << i + kd << ", " << min(i + kd + i3, n) << std::endl;
                        printMatrix2(A12);
                        auto A01 =
                            slice_trapezoid(A, range(i, ib + i),
                                            range(i + ib, min(i + ib + i2, n)));
                        std::cout << "A01 = " << std::endl;
                        printMatrix2(A01);

                        gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                             real_t(-1), A01, work02, real_t(1), A12);
                        std::cout << "gemm (A01, work02)" << std::endl;//////////////

                        auto A22 = slice_trapezoid(
                            A, range(i + kd, min(i + kd + i3, n)),
                            range(i + kd, min(i + kd + i3, n)));
                        std::cout << "A22 = " << std::endl;
                        printMatrix2(A22);

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), work02, real_t(1), A22);
                        std::cout << "herk(work02, A22)" << std::endl;///////////////

                        for (idx_t jj = 0; jj < i3; ++jj) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                A02(ii, jj) = work02(ii, jj);
                            }
                        }
                    }
                }
            }
        }
        else {  // uplo == Lower
            for (idx_t i = 0; i < n; i += nb) {
                idx_t ib;
                if (nb + i < n) {
                    ib = nb;
                }
                else {
                    ib = n - i;
                }

                auto A00 = slice_trapezoid(A, range(i, i + ib),
                                           range(i, min(ib + i, n)));

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
                        auto A10 = slice_trapezoid(
                            A, range(ib + i, ib + i2 + i), range(i, ib + i));

                        trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A10);

                        auto A11 =
                            slice_trapezoid(A, range(ib + i, ib + i2 + i),
                                            range(i + ib, i + ib + i2));

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                             real_t(1), A11);
                    }

                    if (i3 > 0) {
                        auto A10 = slice_trapezoid(
                            A, range(ib + i, ib + i2 + i), range(i, ib + i));

                        auto work20 =
                            slice_trapezoid(work, range(0, i3), range(0, ib));

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0; ii < min(jj + 1, i3); ++ii) {
                                work20(ii, jj) = A(i + kd + ii, jj + i);
                            }
                        }
                        trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                             tlapack::Diag::NonUnit, real_t(1), A00, work20);

                        auto A21 =
                            slice_trapezoid(A, range(kd + i, kd + i + i3),
                                            range(i + ib, i + ib + i2));

                        gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                             real_t(-1), work20, A10, real_t(1), A21);

                        auto A22 =
                            slice_trapezoid(A, range(kd + i, kd + i + i3),
                                            range(kd + i, kd + i + i3));

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                             real_t(1), A22);

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0; ii < min(jj + 1, i3); ++ii) {
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

#endif  // TLAPACK_PBTRF_FULLACCESS_SLICE_TRAPEZOID_HH