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
#include <tlapack/plugins/legacyArray.hpp>
#include <tlapack/LegacyBandedMatrix.hpp>
#include <tlapack/blas/trsm.hpp>
#include <tlapack/lapack/potf2.hpp>
#include <tlapack/blas/herk.hpp>
#include <tlapack/blas/gemm.hpp>
#include <tlapack/lapack/lacpy.hpp>

// <T>LAPACK

// local file

// C++ headers
#include <algorithm>
#include <iomanip>

using namespace tlapack;

template <typename matrix_t>
void printMatrix3(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << std::setw(10) << A(i, j) << " ";
    }
    std::cout << std::endl;
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

#define isSlice_3(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value
template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice_3(SliceSpecRow) && isSlice_3(SliceSpecCol),
                            int>::type = 0>
constexpr auto urinitialsorwhatevauwant(LegacyMatrix<T, idx_t, layout>& A,
                               SliceSpecRow&& rows,
                               SliceSpecCol&& cols) noexcept
{

    idx_t j = cols.first;
    idx_t jj = cols.second - 1;

    idx_t i = rows.first;
    idx_t ii = rows.second - 1;

    idx_t kd = nrows(A);

    idx_t ABrows_first = i % kd;
    idx_t ABrows_second = ii % kd+1; 
    idx_t ABcols_first = j + (i/kd);
    idx_t ABcols_second = jj + (ii/kd)+1;

    idx_t numrows;
    if (ABrows_first > ABrows_second) 
    {
        numrows = kd - ABrows_first + ABrows_second;
        ABcols_second -= 1;
    }
    else
        numrows = ABrows_second - ABrows_first;




    // std::cout << "j = " << j << " jj = " << jj << " i = " << i << " ii = " << ii << " kd = " << kd << std::endl;

    // std::cout << "range (" << ABrows_first << ", " << ABrows_second << ") range (" << ABcols_first << ", " << ABcols_second << ")" << std::endl;
    // std::cout << "ptr: " << ABrows_first + ABcols_first * A.ldim << std::endl;
    // std::cout << "ptr: " << ABrows_second + ABcols_second * A.ldim << std::endl;
    // std::cout << "rows_adder: " << rows_adder << std::endl;
    return LegacyMatrix<T, idx_t, layout>(
        numrows, ABcols_second - ABcols_first,
        (layout == Layout::ColMajor)
            ? &A.ptr[ABrows_first + ABcols_first * A.ldim]
            : &A.ptr[ABrows_first * A.ldim + ABcols_first],
        A.ldim);
}

#undef isSlice_3                    


#define isSlice_ABm1(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value


template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice_ABm1(SliceSpecRow) && isSlice_ABm1(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice_ABm1(LegacyMatrix<T, idx_t, layout>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols) noexcept
{
    idx_t ptr_offset = A.ldim - 1;
    return LegacyMatrix<T, idx_t, layout>(
        rows.second - rows.first, cols.second - cols.first + 1,
        (layout == Layout::ColMajor) ? &A.ptr[ptr_offset + rows.first + cols.first * A.ldim]
                                     : &A.ptr[(ptr_offset + rows.first) * A.ldim + cols.first],
        A.ldim-1);
}

#undef isSlice_ABm1

#define isSlice_ABm1_Lower(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value


template <
    typename T,
    class idx_t,
    Layout layout,
    class SliceSpecRow,
    class SliceSpecCol,
    typename std::enable_if<isSlice_ABm1_Lower(SliceSpecRow) && isSlice_ABm1_Lower(SliceSpecCol),
                            int>::type = 0>
constexpr auto slice_ABm1_Lower(LegacyMatrix<T, idx_t, layout>& A,
                     SliceSpecRow&& rows,
                     SliceSpecCol&& cols) noexcept
{
    idx_t ptr_offset = 0; //A.ldim - 1;
    return LegacyMatrix<T, idx_t, layout>(
        rows.second - rows.first, cols.second - cols.first + 1,
        (layout == Layout::ColMajor) ? &A.ptr[ptr_offset + rows.first + cols.first * A.ldim]
                                     : &A.ptr[(ptr_offset + rows.first) * A.ldim + cols.first],
        A.ldim-1);
}

#undef isSlice_ABm1_Lower

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t kd, size_t nb)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    tlapack::Uplo uplo = tlapack::Uplo::Upper;
    // tlapack::Uplo uplo = tlapack::Uplo::Lower;

    tlapack::Create<matrix_t> new_matrix;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);

    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);

    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    idx_t k = 1;
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            if (i == j) {
                A(i, j) = n * n * n * n;
            }
            else {
                A(i, j) = static_cast<real_t>(k);
            }
                k++;
        }
    }

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j + 1; i < n; i++) {
                A(i, j) = static_cast<real_t>(0);
            }
        }
        for (idx_t j = kd + 1; j < n; j++) {
            for (idx_t i = 0; i < static_cast<int>(j - kd); i++)
                A(i, j) = static_cast<real_t>(0);
        }
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i =
                     std::max(0, static_cast<int>(j) - static_cast<int>(kd));
                 i < j + 1; i++) {
                AB(i + kd - j, j) = A(i, j);
            }
        }
        
    }
    else {
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = 0; i < j; i++) {
                A(i, j) = static_cast<real_t>(0);
            }
        }
        for (idx_t j = kd + 1; j < n; j++) {
            for (idx_t i = 0; i < static_cast<int>(j - kd); i++)
                A(j, i) = static_cast<real_t>(0);
        }
        for (idx_t j = 0; j < n; j++) {
            for (idx_t i = j; i < std::min(static_cast<int>(n),
                                           static_cast<int>(j + kd + 1));
                 i++) {
                AB(i - j, j) = A(i, j);
            }
        }
    }
    lacpy(Uplo::General, A, A_copy);
    std::cout << "A = " << std::endl;
    printMatrix3(A);
    std::cout << "AB = " << std::endl;
    printMatrix3(AB);

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
            auto ABm1 = slice_ABm1(AB, range(0, kd), range(0, m));
            for (idx_t i = 0; i < n; i += nb) {
                // ib = min(nb, n - i)
                idx_t ib;
                if (n < nb + i) {
                    ib = n - i;
                }
                else {
                    ib = nb;
                }
                printMatrix3(ABm1);
                // auto A00 = urinitialsorwhatevauwant(ABm1, range(i, ib + i), range(i, std::min(i + ib, n)));
auto A00 = urinitialsorwhatevauwant(A, range(i, ib + i), range(i, std::min(i + ib, n)));
                potf2(tlapack::Uplo::Upper, A00);

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
                        // auto A01 = urinitialsorwhatevauwant(ABm1, range(i, ib + i),
                        //                  range(i + ib, std::min(i + ib + i2, n)));
auto A01 = urinitialsorwhatevauwant(A, range(i, ib + i),
                                         range(i + ib, std::min(i + ib + i2, n)));

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A01);

                        // auto A11 = urinitialsorwhatevauwant(ABm1, range(i + ib, std::min(i + kd, n)),
                        //                  range(i + ib, std::min(i + kd, n)));
                             auto A11 = urinitialsorwhatevauwant(A, range(i + ib, std::min(i + kd, n)),
                                         range(i + ib, std::min(i + kd, n)));            


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
                        // auto A02 = urinitialsorwhatevauwant(ABm1, range(i, i + ib),
                        //                  range(i + kd, i + kd + i3));
                            auto A02 = urinitialsorwhatevauwant(A, range(i, i + ib),
                                         range(i + kd, i + kd + i3));              
                        auto work02 = urinitialsorwhatevauwant(work, range(0, ib), range(0, i3));

                        for (idx_t jj = 0; jj < i3; jj++)
                            for (idx_t ii = jj; ii < ib; ++ii)
                                work02(ii, jj) = A02(ii, jj);

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, work02);

                        // auto A12 = urinitialsorwhatevauwant(ABm1, range(i + ib, i + kd),
                        //                  range(i + kd, std::min(i + kd + i3, n)));
auto A12 = urinitialsorwhatevauwant(A, range(i + ib, i + kd),
                                         range(i + kd, std::min(i + kd + i3, n)));
                        // auto A01 = urinitialsorwhatevauwant(ABm1, range(i, ib + i),
                        //                  range(i + ib, std::min(i + ib + i2, n)));
auto A01 = urinitialsorwhatevauwant(A, range(i, ib + i),
                                         range(i + ib, std::min(i + ib + i2, n)));
                        gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                             real_t(-1), A01, work02, real_t(1), A12);

                        // auto A22 = urinitialsorwhatevauwant(ABm1, range(i + kd, std::min(i + kd + i3, n)),
                        //                  range(i + kd, std::min(i + kd + i3, n)));
auto A22 = urinitialsorwhatevauwant(A, range(i + kd, std::min(i + kd + i3, n)),
                                         range(i + kd, std::min(i + kd + i3, n)));
                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), work02, real_t(1), A22);
                        std::cout << "herk( work02, A22);" << std::endl;

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
            auto ABm1 = slice_ABm1_Lower(AB, range(0, kd), range(0, m));
            std::cout << "ABm1 = " << std::endl;
            printMatrix3(ABm1);
            for (idx_t i = 0; i < n; i += nb) {
                idx_t ib;
                if (nb + i < n) {
                    ib = nb;
                }
                else {
                    ib = n - i;
                }

                auto A00 = urinitialsorwhatevauwant(ABm1, range(i, i + ib), range(i, std::min(ib + i, n)));
                std::cout << "A00 = " << std::endl;
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
                        auto A10 = urinitialsorwhatevauwant(ABm1, range(ib + i, ib + i2 + i),
                                         range(i, ib + i));
                        std::cout << "A10 = " << std::endl;
                        printMatrix3(A10);

                        trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), A00, A10);

                        auto A11 = urinitialsorwhatevauwant(ABm1, range(ib + i, ib + i2 + i),
                                         range(i + ib, i + ib + i2));

                        std::cout << "A11 = " << std::endl;
                        printMatrix3(A11);
                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                             real_t(1), A11);
                    }

                    if (i3 > 0) {
                        auto A10 = urinitialsorwhatevauwant(ABm1, range(ib + i, ib + i2 + i),
                                         range(i, ib + i));
                        std::cout << "A10 = " << std::endl;
                        printMatrix3(A10);
                    

                        // auto work20 = slice(work, range(0, i3), range(0, ib));

                        // for (idx_t jj = 0; jj < ib; jj++) {
                        //     for (idx_t ii = 0; ii < std::min(jj + 1, i3); ++ii) {
                        //         work20(ii, jj) = A(i + kd + ii, jj + i);
                        //     }
                        // }
                        // trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                        //      tlapack::Diag::NonUnit, real_t(1), A00, work20);

                        auto A21 = urinitialsorwhatevauwant(ABm1, range(kd + i, kd + i + i3),
                                         range(i + ib, i + ib + i2));
                        std::cout << "A21 = " << std::endl;
                        printMatrix3(A21);

                        // gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                        //      real_t(-1), work20, A10, real_t(1), A21);

                        auto A22 = urinitialsorwhatevauwant(ABm1, range(kd + i, kd + i + i3),
                                         range(kd + i, kd + i + i3));
                        std::cout << "A22 = " << std::endl;
                        printMatrix3(A22);

                        // herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                        //      real_t(1), A22);

                        // for (idx_t jj = 0; jj < ib; jj++) {
                        //     for (idx_t ii = 0; ii < std::min(jj + 1, i3); ++ii) {
                        //         A(i + kd + ii, jj + i) = work20(ii, jj);
                        //     }
                        // }
                    }
                }
            }
        }
        potf2(uplo, A_copy);
        std::cout << "Correct A = " << std::endl;
        printMatrix3(A_copy);

        std::cout << "Our A = " << std::endl;
        printMatrix3(A);
    }


//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    std::cout << "Hello World!" << std::endl;

    idx_t m = 13;
    idx_t n = m;
    idx_t kd = 7;
    idx_t nb = 2;

    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    std::cout << std::endl;
    run<float>(m, n, kd, nb);
    printf("-----------------------\n");
    return 0;
}

