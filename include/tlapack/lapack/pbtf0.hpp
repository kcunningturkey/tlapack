#ifndef TLAPACK_PBTF0_HH
#define TLAPACK_PBTF0_HH

namespace tlapack {
template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
int pbtf0(uplo_t uplo, matrix_t& AB)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;

    const idx_t kdp1 = nrows(AB);
    const idx_t n = ncols(AB);
    const idx_t kd = kdp1 - 1;
    const real_t zero(0);

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t i = 0; i < n; ++i) {

           real_t aii = real(AB(kd,i));

            if (aii > zero)
                    AB(kd, i) = sqrt(aii);
            else
            {
                tlapack_error(
                    i + 1,
                    "The leading minor of order j+1 is not positive definite,"
                    " and the factorization could not be completed.");
                return i + 1;
            }

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
            for (idx_t l = kd + 1; l-- > 1;) {  // fix
                for (idx_t j = i + 1 + k_loop; j < min(n, i + kd + 1);
                     ++j) {
                    idx_t row1 = i + kd + 1 + k_loop - j;
                    idx_t col1 = j;

                    idx_t row2 = kd - k_loop - 1;
                    idx_t col2 = i + k_loop + 1;

                    idx_t row3 = k_loop + i + l - j;
                    idx_t col3 = j;

                    AB(row1, col1) -= conj(AB(row2, col2)) * AB(row3, col3);
                }
                ++k_loop;
            }
        }
    }
    else {
        for (idx_t i = 0; i < n; ++i) {
            real_t aii = real(AB(0,i));

            if (aii > zero)
                AB(0, i) = sqrt(aii);
            else
            {
                tlapack_error(
                    i + 1,
                    "The leading minor of order j+1 is not positive definite,"
                    " and the factorization could not be completed.");
                return i + 1;
            }

            // Normalize subdiagonal entries in column i
            for (idx_t j = 1; j < min(kd + 1, n - i); ++j) {
                AB(j, i) /= AB(0, i);
            }

            // Update the trailing submatrix
            for (idx_t l = 0; l < kd; ++l) {
                for (idx_t j = 0; j < min(n - kd, kd - l); ++j) {
                    idx_t col = i + l + 1;
                    if (col + j < n) {
                        AB(j, col) -=
                            AB(j + l + 1, i) * conj(AB(l + 1, i));
                    }
                }
            }
        } 
    }
   return 0; 
}
}  // namespace tlapack

#endif  // TLAPACK_PBTF0_HH