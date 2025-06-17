#ifndef TLAPACK_PBTRF_HH
#define TLAPACK_PBTRF_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {
/// Print matrix A in the standard output
template <typename matrix_t>
void print2Matrix(const matrix_t& A)
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

template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
void pbtrf(uplo_t uplo, matrix_t& AB)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;

    using std::complex;
    using std::cout;
    using std::endl;
    using std::sqrt;
    using std::conj;
    using std::min;

    const idx_t kdp1 = nrows(AB);
    const idx_t n = ncols(AB);
    const idx_t kd = kdp1 - 1;

    for (idx_t i = 0; i < n; ++i) {

        T& aii = AB(kd, i);  // Diagonal entry

        if (real(aii) <= real_t(0)) {
            return;  // Not positive definite
        }

        AB(kd, i) = tlapack::sqrt(aii);  // sqrt on complex is fine

        // Division loop
        idx_t k = i + kd;
        for (idx_t j = 0; j < kd; ++j) {
            if (k < n) {
                AB(j, k) /= AB(kd, i);
            }
            --k;
        }

        // Update trailing submatrix
        idx_t k_loop = 0;
        for (idx_t l = kd + 1; l-- > 1;) { //fix
            for (idx_t j = i + 1 + k_loop; j < std::min(n, i + kd + 1); ++j) {
                idx_t row1 = i + kd + 1 + k_loop - j;
                idx_t col1 = j;

                idx_t row2 = kd - k_loop - 1;
                idx_t col2 = i + k_loop + 1;

                idx_t row3 = k_loop + i + l - j;
                idx_t col3 = j;

                if constexpr (is_complex<T>)
                    AB(row1, col1) -= conj(AB(row2, col2)) * AB(row3, col3);
                else
                    AB(row1, col1) -= AB(row2, col2) * AB(row3, col3);
                // if (row2==row3 && col2 == col3)
                //     AB(row1, col1) -= real(AB(row2, col2)) * real(AB(row2, col2)) + imag(AB(row3, col3)) * imag(AB(row3, col3));
                // else
                    
                // AB(row1, col1) -= complex<T>(real(AB(row2, col2)), (real_t(-1) * imag(AB(row2, col2)))) * AB(row3, col3);
                // AB(row1, col1) = conj(AB(row2, col2));
            }
            ++k_loop;
        }
    }

    // else {
    //     for (idx_t i = 0; i < n; i++){
    //         T aii(0);
    //         aii = AB(0, i);
    //         if (real(aii) <= 0){
    //             return;
    //         }
    //         else {
    //             AB(0, i) = std::sqrt(real(aii));
    //         }
    //         for (idx_t j = 0; std::min(kd + 1, n - i); j++){
    //             AB(j, i) /= AB(0, i);
    //         }
    //     }
    // }
}
}  // namespace tlapack
   // namespace tlapack

#endif