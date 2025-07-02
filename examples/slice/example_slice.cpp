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

// local file

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
            std::cout << std::setw(3) << A(i, j) << " ";
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

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t kd)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    tlapack::Uplo uplo = tlapack::Uplo::Upper;

    tlapack::Create<matrix_t> new_matrix;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);

    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    idx_t k = 1;
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
                A(i, j) = static_cast<real_t>(k);
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
    auto ABm1 = slice_ABm1(AB, range(0, kd), range(0, m));
    // ABm1.ldim -= 1;
    // if (uplo == tlapack::Uplo::Upper) {
    //     for (idx_t j = 0; j < n; j++) {
    //         for (idx_t i =
    //                  std::max(0, static_cast<int>(j) - static_cast<int>(kd));
    //              i <= j; i++) {
    //                 A(i, j) = static_cast<T>(i + 1);  // Only if T is real
    //         }
    //     }
    // }(3, 1)

    std::cout << "A = " << std::endl;
    printMatrix(A);

    std::cout << "AB = " << std::endl;
    printMatrix(AB);

    std::cout << "ABm1 = " << std::endl;
    printMatrix(ABm1);

    auto ABm100 = urinitialsorwhatevauwant(ABm1, range(0, 4), range(0, 4));
    auto A00 = urinitialsorwhatevauwant(A, range(0, 4), range(0, 4));
    std::cout << "ABm100 = " << std::endl;
    printMatrix(ABm100);
    printMatrix(A00);
    auto ABm101 = urinitialsorwhatevauwant(ABm1, range(0, 4), range(5, 8));
    auto A01 = urinitialsorwhatevauwant(A, range(0, 4), range(5, 8));
    std::cout << "ABm101 = " << std::endl;
    printMatrix(ABm101);
    printMatrix(A01);
    auto ABm1_JL = urinitialsorwhatevauwant(ABm1, range(2, 6), range(6, 8));
    auto A_JL = urinitialsorwhatevauwant(A, range(2, 6), range(6, 8));
    std::cout << "ABm1_JL = " << std::endl;
    printMatrix(ABm1_JL);
    printMatrix(A_JL);
    auto ABm1_SH = urinitialsorwhatevauwant(ABm1, range(6, 8), range(6, 9));
    auto A_SH = urinitialsorwhatevauwant(A, range(6, 8), range(6, 9));
    std::cout << "ABm1_SH = " << std::endl;
    printMatrix(ABm1_SH);
    printMatrix(A_SH);
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    std::cout << "Hello World!" << std::endl;

    idx_t m = 11;
    idx_t n = m;
    idx_t kd = 7;

    printf("run< float  >( %d, %d )", static_cast<int>(m), static_cast<int>(n));
    run<float>(m, n, kd);
    printf("-----------------------\n");
    return 0;
}

