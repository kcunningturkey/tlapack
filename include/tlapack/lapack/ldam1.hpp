#ifndef TLAPACK_LDAM1_HH
#define TLAPACK_LDAM1_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf2.hpp"
#include "tlapack/LegacyMatrix.hpp"

namespace tlapack {
/// Print matrix A in the standard output
template <typename matrix_t>
void printdaMatrix(const matrix_t& A)
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

template <typename T>
void ldam1(size_t m1){

    int m = 5, n = 4;

    std::vector<T> A_(m * n);
    for (int i = 0; i < m * n; ++i)
    {
        A_[i] = i + 1;
    }

    LegacyMatrix<T, size_t, Layout::ColMajor> A(m, n, &A_[0], m);

    printdaMatrix(A);

    //     for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         AB(i + kd - j, j) = A(i, j);
    //     }
    // }
}

}

#endif