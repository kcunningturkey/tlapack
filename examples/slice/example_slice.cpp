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
#include <tlapack/lapack/mult_llh.hpp>
#include <tlapack/lapack/mult_uhu.hpp>
#include <tlapack/lapack/lanhe.hpp>

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
            std::cout << std::setw(5) << A(i, j) << " ";
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

#define isSlice_new(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value
template <
typename T,
         class idx_t,
         Layout layout,
         class SliceSpecRow,
         class SliceSpecCol,
         typename std::enable_if<isSlice_new(SliceSpecRow) && isSlice_new(SliceSpecCol),
         int>::type = 0>
         constexpr auto slice_new(LegacyMatrix<T, idx_t, layout>& A,
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

    return LegacyMatrix<T, idx_t, layout>(
            numrows, ABcols_second - ABcols_first,
            (layout == Layout::ColMajor)
            ? &A.ptr[ABrows_first + ABcols_first * A.ldim]
            : &A.ptr[ABrows_first * A.ldim + ABcols_first],
            A.ldim);
}
#undef isSlice_new                    


#define isSLice_ABm1_Upper(SliceSpec) !std::is_convertible<SliceSpec, idx_t>::value
template <
typename T,
         class idx_t,
         Layout layout,
         class SliceSpecRow,
         class SliceSpecCol,
         typename std::enable_if<isSLice_ABm1_Upper(SliceSpecRow) && isSLice_ABm1_Upper(SliceSpecCol),
         int>::type = 0>
         constexpr auto slice_ABm1(LegacyMatrix<T, idx_t, layout>& A,
                 SliceSpecRow&& rows,
                 SliceSpecCol&& cols) noexcept
{
    idx_t ptr_offset = A.ldim - 1;

    idx_t kd = nrows(A) - 1;
    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;

    return LegacyMatrix<T, idx_t, layout>(
            rows.second - rows.first, numcols,
            (layout == Layout::ColMajor) ? &A.ptr[ptr_offset + rows.first + cols.first * A.ldim]
            : &A.ptr[(ptr_offset + rows.first) * A.ldim + cols.first],
            A.ldim - 1);
}
#undef isSLice_ABm1_Upper

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
    idx_t ptr_offset = 0;

    idx_t kd = nrows(A) - 1;
    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;

    return LegacyMatrix<T, idx_t, layout>(
            rows.second - rows.first, numcols,
            (layout == Layout::ColMajor) ? &A.ptr[ptr_offset + rows.first + cols.first * A.ldim]
            : &A.ptr[(ptr_offset + rows.first) * A.ldim + cols.first],
            A.ldim - 1);
}
#undef isSlice_ABm1_Lower

/// @brief Options struct for pbtrf()
struct BlockedBandedCholeskyOpts : public EcOpts {
    constexpr BlockedBandedCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts) {};

    size_t nb = 32;  ///< Block size
};


template<typename uplo_t, typename matrix_t>
void pbtrf(uplo_t uplo, matrix_t& A, size_t n, const BlockedBandedCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;
    using real_t = tlapack::real_type<T>;

    tlapack::Create<matrix_t> new_matrix;

    const idx_t nb = opts.nb;

    idx_t kd = nrows(A);
    //idx_t n = ncols(A); // THIS IS INCORRECT BECAUSE WE ADD COLUMNS

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
        // auto ABm1 = slice_ABm1(AB, range(0, kd), range(0, m));
        for (idx_t i = 0; i < n; i += nb) {
            // ib = min(nb, n - i)
            idx_t ib;
            if (n < nb + i) {
                ib = n - i;
            }
            else {
                ib = nb;
            }

            //std::cout << "n = " << n << ", i = " << i << ", nb = " << nb << ", ib = " << ib << std::endl;
            // auto A00 = slice_new(CD, range(i, ib + i), range(i, std::min(i + ib, n)));
            auto A00 = slice_new(A, range(i, min(ib + i, n)), range(i, std::min(i + ib, n)));
    //std::cout << "HENC" << std::endl;
    //printMatrix(A);
    //printMatrix(A00);


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
                     //auto A01 = slice_new(CD, range(i, ib + i),
                                      //range(i + ib, std::min(i + ib + i2, n)));
                    auto A01 = slice_new(A, range(i, ib + i),
                            range(i + ib, std::min(i + ib + i2, n)));

                    trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, A01);

                     //auto A11 = slice_new(CD, range(i + ib, std::min(i + kd, n)),
                                      //range(i + ib, std::min(i + kd, n)));
                    auto A11 = slice_new(A, range(i + ib, std::min(i + kd, n)),
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
                     //auto A02 = slice_new(CD, range(i, i + ib),
                                      //range(i + kd, i + kd + i3));
                    auto A02 = slice_new(A, range(i, i + ib),
                            range(i + kd, i + kd + i3));              

                    auto work02 = slice_new(work, range(0, ib), range(0, i3));

                    for (idx_t jj = 0; jj < i3; jj++)
                        for (idx_t ii = jj; ii < ib; ++ii)
                            work02(ii, jj) = A02(ii, jj);

                    trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, work02);

                    //auto A12 = slice_new(CD, range(i + ib, i + kd),
                                      //range(i + kd, std::min(i + kd + i3, n)));
                    auto A12 = slice_new(A, range(i + ib, i + kd),
                            range(i + kd, std::min(i + kd + i3, n)));

                    //auto A01 = slice_new(CD, range(i, ib + i),
                                      //range(i + ib, std::min(i + ib + i2, n)));
                    auto A01 = slice_new(A, range(i, ib + i),
                            range(i + ib, std::min(i + ib + i2, n)));

                    gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                            real_t(-1), A01, work02, real_t(1), A12);

                    //auto A22 = slice_new(CD, range(i + kd, std::min(i + kd + i3, n)),
                                      //range(i + kd, std::min(i + kd + i3, n)));
                    auto A22 = slice_new(A, range(i + kd, std::min(i + kd + i3, n)),
                            range(i + kd, std::min(i + kd + i3, n)));

                    herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                            real_t(-1), work02, real_t(1), A22);

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
        
        // auto ABm1 = slice_ABm1_Lower(A, range(0, kd), range(0, n));
        // std::cout << "here" << std::endl;
        //std::cout << "ABm1 = " << std::endl;
        //printMatrix(ABm1);
        for (idx_t i = 0; i < n; i += nb) {
            idx_t ib;
            if (nb + i < n) {
                ib = nb;
            }
            else {
                ib = n - i;
            }

            // std::cout << "here" << std::endl;
            // auto A00 = slice_new(ABm1, range(i, i + ib), range(i, std::min(ib + i, n)));
            auto A00 = slice_new(A, range(i, i + ib), range(i, std::min(ib + i, n)));
            // std::cout << "A00 = " << std::endl;
            // printMatrix(A00);

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
                    // auto A10 = slice_new(ABm1, range(ib + i, ib + i2 + i),
                    auto A10 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i, ib + i));
                    // std::cout << "A10 = " << std::endl;
                    // printMatrix(A10);

                    trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                            tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                            real_t(1), A00, A10);

                    // auto A11 = slice_new(ABm1, range(ib + i, ib + i2 + i),
                    auto A11 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i + ib, i + ib + i2));

                    // std::cout << "A11 = " << std::endl;
                    // printMatrix(A11);
                    herk(uplo, tlapack::Op::NoTrans, real_t(-1), A10,
                            real_t(1), A11);
                }

                if (i3 > 0) {
                    // auto A10 = slice_new(ABm1, range(ib + i, ib + i2 + i),
                    auto A10 = slice_new(A, range(ib + i, ib + i2 + i),
                            range(i, ib + i));
                    // std::cout << "A10 = " << std::endl;
                    // printMatrix(A10);

                    // auto A20 = slice_new(ABm1, range(ib + i, ib + i2 + i), range(i + ib, i + ib + i2));
                    auto A20 = slice_new(A, range(kd + i, min(kd + i3 + i, n)), range(i, i + ib));
                    // std::cout << "A20 =  " << std::endl;
                    // printMatrix(A20);

                    auto work20 = slice(work, range(0, i3), range(0, ib));

                    for (idx_t jj = 0; jj < ib; jj++) {
                        for (idx_t ii = 0; ii < std::min(jj + 1, i3); ++ii) {
                            work20(ii, jj) = A20(ii, jj);
                        }
                    }
                    // std::cout << "work = " << std::endl;
                    // printMatrix(work20);
                    trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                         tlapack::Diag::NonUnit, real_t(1), A00, work20);

                    // auto A21 = slice_new(ABm1, range(kd + i, kd + i + i3),
                    auto A21 = slice_new(A, range(kd + i, kd + i + i3),
                            range(i + ib, i + ib + i2));
                    // std::cout << "A21 = " << std::endl;
                    // printMatrix(A21);

                    gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                         real_t(-1), work20, A10, real_t(1), A21);

                    // auto A22 = slice_new(ABm1, range(kd + i, kd + i + i3),
                    auto A22 = slice_new(A, range(kd + i, kd + i + i3),
                            range(kd + i, kd + i + i3));
                    // std::cout << "A22 = " << std::endl;
                    // printMatrix(A22);

                    herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                         real_t(1), A22);

                    for (idx_t jj = 0; jj < ib; jj++) {
                        for (idx_t ii = 0; ii < std::min(jj + 1, i3); ++ii) {
                            A20(ii, jj) = work20(ii, jj);
                        }
                    }
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
    template <typename T>
void run(size_t m, size_t n, size_t kd, size_t nb)
{
    using real_t = tlapack::real_type<T>;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = std::pair<idx_t, idx_t>;

    // tlapack::Uplo uplo = tlapack::Uplo::Upper;
    tlapack::Uplo uplo = tlapack::Uplo::Lower;

    tlapack::Create<matrix_t> new_matrix;

    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);

    std::vector<T> A_copy_;
    auto A_copy = new_matrix(A_copy_, m, n);

    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    std::vector<T> CD_;
    idx_t numcols = ((kd + 1) * ncols(A) - 1) / kd;
    auto CD = new_matrix(CD_, kd, numcols);

    // for (idx_t i = 0; i < kd + 1; i++) {
    //     for (idx_t j = 0; j < n; j++) {
    //         AB(i, j) = static_cast<real_t>(0xCAFEBABE);
    //     }
    // }
    for (idx_t i = 0; i < kd; i++) {
        for (idx_t j = 0; j < numcols; j++) {
            CD(i, j) = static_cast<real_t>(0xCAFEBABE);
        }
    }

    idx_t k = 1;
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < n; ++i) {
            //A(i, j) = static_cast<real_t>(k);
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
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j + 1; i < n; i++)
                A(i, j) = static_cast<real_t>(0);

        for (idx_t j = kd + 1; j < n; j++)
            for (idx_t i = 0; i < static_cast<int>(j - kd); i++)
                A(i, j) = static_cast<real_t>(0);

        for (idx_t j = 0; j < n; j++)
            for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd)); i < j + 1; i++)
                AB(i + kd - j, j) = A(i, j);

        // create the upper blocked compressed martix
        idx_t jj = 0;  // push one column over to make room for the diagonal
        for (idx_t k = 0; k < n; k += kd)
        {
            // diagonal blocks
            for (idx_t i = 0; i < min(n-k, kd); i++)
            {
                idx_t i2;
                if (n > k)
                    i2 = n - k;
                else
                    i2 = 0;

                for (idx_t j = i; j < min(kd, i2); j++) {
                    CD(i % kd, k + j + jj) = A(k + i, k +j);
                }
            }

            // off diagonal blocks
            for (idx_t i = 0; i < min(n - k, kd); i++)
            {
                idx_t i2;
                if (n > k + kd)
                    i2 = n - k - kd;
                else
                    i2 = 0;

                for (idx_t j = 0; j < min(i + 1, i2); j++) {
                    CD(i % kd, k + j + jj + kd) = A(k + i, k + j + kd);
                }
            }
            jj++;
        }

    }
    else {
        idx_t jj = 0;
        std::cout << "CD = " << kd << " x " << numcols << std::endl;
        for (idx_t j = 0; j < n; j++)
            for (idx_t i = 0; i < j; i++)
                A(i, j) = static_cast<real_t>(0);

        for (idx_t j = kd + 1; j < n; j++)
            for (idx_t i = 0; i < static_cast<int>(j - kd); i++)
                A(j, i) = static_cast<real_t>(0);

        for (idx_t j = 0; j < n; j++)
            for (idx_t i = j; i < std::min(static_cast<int>(n), static_cast<int>(j + kd + 1)); i++)
                AB(i - j, j) = A(i, j);

        for (idx_t k = 0; k < n; k += kd) {
            // std::cout << "in k loop = " << k << std::endl;
            // diagonal triangles
            for (idx_t i = 0; i < min(n-k, kd); i++) {
                for (idx_t j = 0; j < i+1; j++) {
                    CD(i % kd, k + j + jj) = A(k + i, k +j);
                }
            }
            
            for (idx_t i = 0; i < min(n-k-kd, kd); i++) {
                // std::cout << "in i loop = " << i << std::endl;
                // idx_t i2;
                // if (n > k + kd)
                //     i2 = k + kd + 1;
                // else
                //     i2 = 0;
                for (idx_t j = k + 1; j < k+kd+1-i; ++j) {
                    // std::cout << "in j loop = "<< j << std::endl;
                    std::cout << "CD(" << i % kd << ", " << i + j + jj << ") = A(" << k + kd + i << ", " << j << ")" << std::endl;
                    // CD(i % kd, k + j + i) = A(k + kd + i, k + j + jj + i - 1);
                    }
            }


            jj += 1;
        }
    }
// auto ABm1 = slice_ABm1(AB, range(0, kd), range(0, m));
auto ABm1 = slice_ABm1_Lower(AB, range(0, kd), range(0, m));
std::cout << "ABm1 = " << std::endl;
printMatrix(ABm1);

    lacpy(Uplo::General, A, A_copy);
    // std::cout << "A = " << std::endl;
    // printMatrix(A);
    // std::cout << "CD = " << std::endl;
    // printMatrix(CD);
    // std::cout << "ABm1 = " << std::endl;
    // printMatrix(ABm1);

    // Check each type of data storage format

    // potf2(uplo, A_copy);
    // std::cout << "Correct A = " << std::endl;
    // printMatrix(A_copy);


    BlockedBandedCholeskyOpts opts;
    opts.nb = nb;

    // Full access
    real_t normAbefore = lanhe(tlapack::Norm::Fro, uplo, A);
    pbtrf(uplo, A, n, opts);
    // std::cout << "A = " << std::endl;
    // printMatrix(A);

    // Compressed Blocks
    pbtrf(uplo, CD, n, opts);
    // std::cout << "CD = " << std::endl;
    // printMatrix(CD);

    // LAPACK AB format which is the same as ABm1
    pbtrf(uplo, ABm1, n, opts);
    // std::cout << "ABm1 = " << std::endl;
    // printMatrix(ABm1);

    std::cout << "checking" << std::endl;

    //---------------------------------------------------------------------------norm test----------------------------------------

    std::vector<T> CD_copy_;
    auto CD_copy = new_matrix(CD_copy_, m, n);

    std::vector<T> ABm1_copy_;
    auto ABm1_copy = new_matrix(ABm1_copy_, m, n);

    for (idx_t i = 0; i < m; ++i) {
        for (idx_t j = 0; j < n; ++j) {
            CD_copy(i, j) = static_cast<real_t>(0);
            ABm1_copy(i, j) = static_cast<real_t>(0);
        }
    }

    if (uplo == Uplo::Upper) {
        idx_t jj = 0;  // push one column over to make room for the diagonal
        for (idx_t k = 0; k < n; k += kd)
        {
            // diagonal blocks
            for (idx_t i = 0; i < min(n-k, kd); i++)
            {
                idx_t i2;
                if (n > k)
                    i2 = n - k;
                else
                    i2 = 0;

                for (idx_t j = i; j < min(kd, i2); j++) {
                    CD_copy(k + i, k +j) = CD(i % kd, k + j + jj);
                    ABm1_copy(k + i, k +j) = ABm1(i % kd, k + j + jj);
                }
            }

            // off diagonal blocks
            for (idx_t i = 0; i < min(n - k, kd); i++)
            {
                idx_t i2;
                if (n > k + kd)
                    i2 = n - k - kd;
                else
                    i2 = 0;

                for (idx_t j = 0; j < min(i + 1, i2); j++) {
                    CD_copy(k + i, k + j + kd) = CD(i % kd, k + j + jj + kd);
                    ABm1_copy(k + i, k + j + kd) = ABm1(i % kd, k + j + jj + kd);
                }
            }
            jj++;
        }
    //     std::cout << "correct A = " << std::endl;
    // printMatrix(A_copy);
    // std::cout << "CD_copy = " << std::endl;
    // printMatrix(CD_copy);
    // std::cout << "ABm1_copy = " << std::endl;
    // printMatrix(ABm1_copy);

    mult_uhu(CD_copy);
    mult_uhu(ABm1_copy);
    mult_uhu(A);
    }
    else {
        idx_t jj = 0; 
        for (idx_t k = 0; k < n; k += kd) {

            // diagonal triangles
            for (idx_t i = 0; i < min(n-k, kd); i++) {
                for (idx_t j = 0; j < i+1; j++) {
                    // ;
                    CD_copy(k + i, k +j) = CD(i % kd, k + j + jj);
                    ABm1_copy(k + i, k +j) = ABm1(i % kd, k + j + jj);
                }
            }

            for (idx_t i = 0; i < min(n-k, kd - 1); i++) {
                idx_t i2;
                if (n > k + kd)
                    i2 = k + kd + 1;
                else
                    i2 = 0;
                for (idx_t j = k + 1; j < min(k+kd+1, i2); ++j) {
                    CD_copy(k + kd + i, k + j + jj + i - 1) = CD(i % kd, k + j + jj+ i);
                    ABm1_copy(k + kd + i, k + j + jj + i - 1) = ABm1(i % kd, k + j + jj+ i);
                    }
            }


            jj += 1;
        }
    //     std::cout << "correct A = " << std::endl;
    // printMatrix(A_copy);
    // std::cout << "CD_copy = " << std::endl;
    // printMatrix(CD_copy);
    // std::cout << "ABm1_copy = " << std::endl;
    // printMatrix(ABm1_copy);

    mult_llh(CD_copy);
    mult_llh(ABm1_copy);
    mult_llh(A);
    }

    // std::cout << "A = " << std::endl;
    // printMatrix(A);

    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = 0; i < n; i++) {
            CD_copy(i, j) -= A_copy(i, j);
            A(i, j) -= A_copy(i, j);
            ABm1_copy(i, j) -= A_copy(i, j);
        }
    }

    real_t normCD = lanhe(Norm::Fro, uplo, CD_copy);
    real_t normA = lanhe(Norm::Fro, uplo, A);
    real_t normABm1 = lanhe(Norm::Fro, uplo, ABm1_copy);

    std::cout << std::endl;

    std::cout << "The norm of A = " << normA/normAbefore << std::endl;
    std::cout << "The norm of CD = " << normCD/normAbefore << std::endl;
    std::cout << "The norm of ABm1 = " << normABm1/normAbefore << std::endl;

    std::cout << std::endl;

    //  NOTE:  Only upper triangular is implemented.  Lower triangular has not
    //  been started yet.  First get the storage for the Compressed
    //  Block format correct, second get the full access code working, then we 
    //  should be able to apply the full access code to each of the other formats.
}


//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    using std::size_t;

    using idx_t = size_t;

    idx_t m = 13;
    idx_t n = m;
    idx_t kd = 5;
    idx_t nb = 2;

    printf("run< float  >( %d, %d )\n", static_cast<int>(m), static_cast<int>(n));
    std::cout << std::endl;
    run<float>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )\n", static_cast<int>(m), static_cast<int>(n));
    run<double>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<long double>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<float> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<float>>(m, n, kd, nb);
    printf("-----------------------\n");

    printf("run< complex<double> >( %d, %d )\n", static_cast<int>(m),
           static_cast<int>(n));
    run<std::complex<double>>(m, n, kd, nb);
    printf("-----------------------\n");

    return 0;
}