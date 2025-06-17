#ifndef TLAPACK_PBSTF_HH
#define TLAPACK_PBSTF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/potrf.hpp"
#include "tlapack/lapack/pttrf.hpp"

// C++ Includes
#include <algorithm>

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
// Do we need ldab or info?
auto pbstf(uplo_t uplo, matrix_t& AB)
{
    std::cout << std::endl << std::endl << "pbstf called" << std::endl;
    using T = tlapack::type_t<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using std::size_t;
    using idx_t = tlapack::size_type<matrix_t>;
    tlapack::Create<matrix_t> new_matrix;

    bool verbose = true;

    // Check arguments
    tlapack_check_false(uplo != tlapack::Uplo::Lower &&
                        uplo != tlapack::Uplo::Upper);

    // Define variables for pbstf
    size_t m = nrows(AB);

    size_t n = ncols(AB);

    idx_t kd = nrows(AB) - 1;

    // Checks if AB is Upper Triangular
    if (uplo == tlapack::UPPER_TRIANGLE) {
        std::cout << std::endl << "AB is Upper Triangular";

        // Checks for appropriate algorithm depending on band depth size
        if (kd == 0) {
            for (idx_t j = 0; j < n; j++) {
                const real_t ajj = std::real(AB(kd, j));
                AB(kd, j) = sqrt(ajj);
            }
            std::cout << std::endl << "Updated AB=";
            print2Matrix(AB);
        }
        else if (kd == 1) {
            // Create the vector D to pass to pttrf
            std::vector<T> D_;
            auto D = new_matrix(D_, 1, n);
            for (idx_t j = 0; j < n; ++j)
                D(0, j) = AB(kd, j);

            std::vector<T> E_;
            auto E = new_matrix(E_, 1, n - 1);
            for (idx_t j = 0; j < n - 1; ++j) {
                E(0, j) = AB(0, j + 1);
                // For pttrf subdiagonals are needed
                E(0, j) = std::conj(E(0, j));
            }
            // std::cout << std::endl << "Vector D=";
            // print2Matrix(D);
            // std::cout << std::endl << "Vector E=";
            // print2Matrix(E);

            tlapack::pttrf(D_, E_);

            for (idx_t j = 0; j < n - 1; j++) {
                AB(kd, j) = D(0, j);
                AB(0, j + 1) = E(0, j);
            }

            std::cout << std::endl << "Factor D=";
            print2Matrix(D);
            std::cout << std::endl << "Factor L=";
            print2Matrix(E);

            // std::cout << std::endl << "Updated AB=";
            // print2Matrix(AB);
        }
        else if (kd == n - 1) {
            // Turning AB back into A so that we may call potrf
            std::vector<T> Aprime_;
            auto Aprime = new_matrix(Aprime_, m, n);
            std::cout << std::endl << "Calling kd == n -1";
            for (idx_t j = 0; j < n; j++) {
                for (idx_t i = kd + 1;
                     i-- > std::max(0, static_cast<int>(kd - j));) {
                    Aprime(i - kd + j, j) = AB(i, j);
                }
            }
            // std::cout << std::endl << "Aprime=";
            // print2Matrix(Aprime);

            potrf(tlapack::UPPER_TRIANGLE, Aprime);

            std::cout << std::endl << "portf Aprime=";
            print2Matrix(Aprime);

            for (idx_t j = 0; j < n; j++)
                for (idx_t i = 0; i <= j; i++)
                    AB(i + kd - j, j) = Aprime(i, j);

            std::cout << std::endl << "AB post potrf=";
            print2Matrix(AB);
        }
        else  ////////////
            // std::cout << std::endl << "Calling blocked pbstf";
            // idx_t M = (n + kd) / 2;
            // std::cout << std::endl << "M=" << M;

            
            std::cout << std::endl << "AB=";
            print2Matrix(AB);
            for (idx_t j = 0; j < 1; ++j) {
                idx_t k = kd + j;
                auto factor = std::complex<float>(1.0f) / std::sqrt(AB(kd, j));

                for (idx_t i = 0; i < kd; ++i) {
                    if (k < n){
                    AB(i, k) = factor * AB(i, k);  // Update element       
                    }                    // Move one superdiagonal up
                k--;
                }

            }
        std::cout << std::endl << "AB ";
        print2Matrix(AB);
    }
    else if (uplo == tlapack::LOWER_TRIANGLE) {
        std::cout << std::endl << "AB is Lower Triangular";
    }
    else {
        std::cout
            << std::endl
            << "Invalid matrix shape. AB must be upper or lower triangular"
            << std::endl;
    }
    if (verbose == true) {
        std::cout << std::endl;
        std::cout << std::endl << "Cols=" << n;
        std::cout << std::endl << "Rows=" << m;
        std::cout << std::endl << "Kd=" << kd;
    }
}
#endif  // TLAPACK_PBSTF_HH
