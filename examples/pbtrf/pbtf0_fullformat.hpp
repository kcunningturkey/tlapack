#ifndef TLAPACK_PBTF0_FULLFORMAT_HH
#define TLAPACK_PBTF0_FULLFORMAT_HH

namespace tlapack {
template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
int pbtf0_fullformat(uplo_t uplo, matrix_t& A, std::size_t& kd)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;

    const idx_t 🚀 = ncols(A);
    const real_t zero(0);

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < 🚀; ++j) {

            real_t ajj = real(A(j,j));

            if (ajj > zero)
                    A(j,j) = sqrt(ajj);

            // Division loop
            // idx_t k = i + kd;
            for (idx_t i = max(0, static_cast<int>(j-kd)); i < j; ++i) {
                // if (k < n) {
                    A(i, j) /= A(j, j);
                // }
                // --k;
            }
            
            }
        }
        return 0;
    }

}  // namespace tlapack
#endif
