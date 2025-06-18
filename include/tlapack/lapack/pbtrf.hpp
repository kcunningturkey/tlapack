#ifndef TLAPACK_PBTRF_HH
#define TLAPACK_PBTRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"

namespace tlapack {
/// Print matrix A in the standard output
template <typename matrix_t>
void printaMatrix(const matrix_t& A)
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

template <typename matrixA_t, typename matrixB_t>
void trsm_b(matrixA_t& A, matrixB_t& B, std::size_t kd)
{
    // only for upper, conjtrans, left
    using TA = tlapack::type_t<matrixA_t>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using scalar_t = scalar_type<TA>;

    const idx_t n = ncols(A);
    const idx_t m = ncols(B);

    for (idx_t j = 0; j < m; ++j) {
        for (idx_t i = n - kd + j; i < n; ++i) {
            scalar_t sum = B(i, j);
            for (idx_t k = 0; k < i; ++k)
                sum -= conj(A(k, i)) * B(k, j);
            sum /= conj(A(i, i));
            B(i, j) = sum;
            // std::cout << std::endl << i << ", " << j << std::endl;
        }
    }
}

template <typename matrixA_t, typename matrixC_t>
void herk_b(matrixA_t& A, matrixC_t& C, std::size_t kd)
{
    using TA = tlapack::type_t<matrixA_t>;
    using idx_t = tlapack::size_type<matrixA_t>;
    using real_t = tlapack::real_type<TA>;

    idx_t k = nrows(A);
    real_t beta = 1;
    real_t alpha = -1;

    for (idx_t j = 0; j < kd; ++j) {
        for (idx_t i = 0; i < j; ++i) {
            TA sum(0);
            for (idx_t l = k - kd + j; l < k; ++l){
                sum += conj(A(l, i)) * A(l, j);
            }
            C(i, j) = alpha * sum + beta * C(i, j);
        }
        real_type<TA> sum(0);
        for (idx_t l = k - kd; l < k; ++l)
            sum +=
                real(A(l, j)) * real(A(l, j)) + imag(A(l, j)) * imag(A(l, j));
        C(j, j) = alpha * sum + beta * real(C(j, j));
    }
}
template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
void pbtrf(uplo_t uplo, matrix_t& A, std::size_t kd)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using range = pair<idx_t, idx_t>;

    using std::complex;
    using std::conj;
    using std::cout;
    using std::endl;
    using std::min;
    using std::sqrt;

    const idx_t n = ncols(A);
    const idx_t n0 = n / 2;

    auto A00 = slice(A, range(0, n0), range(0, n0));
    auto A01 = slice(A, range(0, n0), range(n0, n));
    auto A11 = slice(A, range(n0, n), range(n0, n));
    std::cout << "\ndone slices" << std::endl;

    if (kd >= n/2) {
        potrf(uplo, A00);
        trsm(Side::Left, uplo, Op::ConjTrans, Diag::NonUnit, real_t(1), A00, A01);
        herk(uplo, Op::ConjTrans, real_t(-1), A01, real_t(1), A11);
        potrf(uplo, A11);
    }
    else {
    // std::cout << "slice A00 = " << std::endl;
    // printaMatrix(A00);

    // std::cout << "\nslice A01 = " << std::endl;
    // printaMatrix(A01);

    // std::cout << "\nslice A11 = " << std::endl;
    // printaMatrix(A11);

    pbtrf(uplo, A00, kd);
    std::cout << "\nrecursion done" << std::endl;

    trsm_b(A00, A01, kd);
    std::cout << "trsm done" << std::endl;

    // std::cout << "\nA01 = " << std::endl;
    // printaMatrix(A01);

    herk_b(A01, A11, kd);

    pbtrf(uplo, A11, kd);
    }
    // std::cout << "\nA11 = " << std::endl;
    // printaMatrix(A11);
}
}  // namespace tlapack
   // namespace tlapack

#endif