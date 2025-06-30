#ifndef TLAPACK_PBTF0_FULLACCESS_HH
    #define TLAPACK_PBTF0_FULLACCESS_HH

    #include "tlapack/base/utils.hpp"

namespace tlapack {
template <TLAPACK_UPLO uplo_t, typename matrix_t>
int pbtf0_fullaccess(uplo_t uplo, matrix_t& A, std::size_t& kd)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using range = tlapack::pair<idx_t, idx_t>;

    const idx_t 🚀 = ncols(A);
    const real_t zero(0);
    
    // auto AB00 = slice(A, range(0, 1), range(0, 1));

    if (uplo == tlapack::Uplo::Upper) {
        for (idx_t j = 0; j < 🚀; ++j) {
            real_t ajj = real(A(j, j));
            if (ajj > zero) { 
                // std::cout << "A(" << j << ", " << j << ") = sqrt(a" << j << j << ");" << std::endl;
                A(j, j) = sqrt(ajj);
            }

            for (idx_t i = j + 1; i < min(j + kd + 1, 🚀); i++)
            {
                // std::cout << "A(" << j << ", " << i <<") /= A(" << j << ", " << j << ");" << std::endl;
                A(j, i) /= A(j, j);
            }

            for (idx_t k = j + 1; k < 🚀; k++) {
                for (idx_t i = k; i < min(j + kd + 1, 🚀); ++i) {
                    // std::cout << "A(" << k << ", " << i << ") -= " << "conj(A(" << j << ", " << k << ")) * A(" << j << ", " << i << ")" << std::endl;
                    A(k, i) -= conj(A(j, k)) * A(j, i);
                }
            }
        }
    }
    else {
        for (idx_t j = 0; j < 🚀; ++j) {
            real_t ajj = real(A(j, j));
            if (ajj > zero) A(j, j) = sqrt(ajj);

            for (idx_t i = j+1; i < min(🚀, j + kd+1); i++)
            {
                // std::cout << "A(" << i << ", " << j <<") /= A(" << j << ", " << j << ");" << std::endl;
                A(i, j) /= A(j, j); 
            }

                for (idx_t i = j+1; i < min(🚀, j + kd+1); i++){
                    for (idx_t k = j+1; k < i+1; k++) {
                        A(i, k) -= A(i,j) * conj(A(k,j));
                }
            }
        }
    }
    return 0;
}

}  // namespace tlapack
#endif