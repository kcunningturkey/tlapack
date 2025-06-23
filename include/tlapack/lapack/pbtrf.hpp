#ifndef TLAPACK_PBTRF_HH
#define TLAPACK_PBTRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf2.hpp"

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
void trsm_squeesh(matrixA_t& A, matrixB_t& B, std::size_t kd)
{
    // // only for upper, conjtrans, left
    // using TA = tlapack::type_t<matrixA_t>;
    // using idx_t = tlapack::size_type<matrixA_t>;
    // using scalar_t = scalar_type<TA>;

    // const idx_t m = nrows(B);
    // const idx_t n = ncols(B);

    // for (idx_t j = 0; j < n; ++j) {
    //     for (idx_t i = 0; i < m; ++i) {
    //         scalar_t sum = B(i, j);
    //         for (idx_t k = 0; k < i; ++k){
    //             idx_t l = n - 1;
    //             sum -= conj(A(l, k)) * B(k, j);
    //         }
    //         --l;
    //         B(i, j) = sum;
    //     }
    // }

    // const idx_t m = ncols(A);
    // const idx_t n = ncols(B);
    // std::cout << "\nA = " << std::endl;
    // printaMatrix(A);
    // std::cout << "\nB = " << std::endl;
    // printaMatrix(B);
    // if (n % 2 == 0) {
    //     std::cout << "option even" << std::endl;
    //     for (idx_t j = 0; j < n; ++j) {
    //         for (idx_t i = std::max(static_cast<int>(0),
    //                                 static_cast<int>(m - kd + j));
    //              i < m; ++i) {
    //             scalar_t sum = B(i, j);
    //             std::cout << "\nstart B(" << i << ", " << j << ") = " << B(i,
    //             j)
    //                       << std::endl;
    //             for (idx_t k = std::max(static_cast<int>(0),
    //                                     static_cast<int>(m - kd + j));
    //                  k < i; ++k) {
    //                 sum -= conj(A(k, i)) * B(k, j);
    //                 std::cout << "\nconjA = " << conj(A(k, i))
    //                           << " B = " << B(k, j) << std::endl;
    //                 std::cout << "j = " << j << " i = " << i << " k = " << k
    //                           << " the sum = " << sum << std::endl;
    //             }
    //             sum /= conj(A(i, i));
    //             B(i, j) = sum;
    //             std::cout << "B(" << i << ", " << j << ")" << " = " << sum
    //                       << std::endl;
    //         }
    //     }
    // }
}
template<typename matrix_t>
void pbtrf_block_ldim(matrix_t& AB, std::size_t kd)
{
    using idx_t = tlapack::size_type<matrix_t>;
    using range = pair<idx_t, idx_t>;

    idx_t IB = kd/2;
    
    auto temp = &AB.ptr[0];
    AB.ldim -= 1;
    AB.ptr = &AB.ptr[kd];
    potf2(Uplo::Upper, AB);
    AB.ptr = temp;
    AB.ldim += 1;
}


template <typename matrix_t>
void pbtrf_cheat_squish(matrix_t& AB, std::size_t kd)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using range = pair<idx_t, idx_t>;
    tlapack::Create<matrix_t> new_matrix;

    auto AB00 = slice(AB, range(7, 20), range(0, 13));

    // std::cout << "\nA00 = " << std::endl;
    // printaMatrix(A00);
    // std::cout << "\nA01 = " << std::endl;
    // printaMatrix(A01);
    // std::cout << "\nA11 = " << std::endl;
    // printaMatrix(A11);

    idx_t n = ncols(AB00);
    idx_t k = ncols(AB);

    pbtf2(Uplo::Upper, AB00);

    std::cout << "\nAB00 = " << std::endl;
    printaMatrix(AB00);

    std::vector<T> A_;
    auto A = new_matrix(A_, k, k);

    // full
    for (idx_t j = 0; j < k; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            A(i, j) = AB(i + kd - j, j);
        }
    }

    auto A00 = slice(A, range(0, 13), range(0, 13));
    auto A01 = slice(A, range(0, 13), range(13, 20));
    auto A11 = slice(A, range(13, 20), range(13, 20));

    std::cout << "\nA00 = " << std::endl;
    printaMatrix(A00);

    std::cout << "\nA01 = " << std::endl;
    printaMatrix(A01);

    std::cout << "\nA11 = " << std::endl;
    printaMatrix(A11);

    trsm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1), A00,
         A01);
    std::cout << "\ndone trsm" << std::endl;

    herk(Uplo::Upper, Op::ConjTrans, real_t(-1), A01, real_t(1), A11);
    std::cout << "done herk" << std::endl;

    idx_t m = ncols(A11);

    std::vector<T> AB11_;
    auto AB11 = new_matrix(AB11_, m, m);

    for (idx_t j = 0; j < m; j++) {
        for (idx_t i =
                 std::max(0, static_cast<int>(j) - static_cast<int>(m - 1));
             i < j + 1; i++) {
            std::cout << "AB11(" << i << " + " << m - 1 << " - " << j << ", "
                      << j << ") = A11(" << i << ", " << j << ")" << std::endl;
            AB11(i + m - 1 - j, j) = A11(i, j);
        }
    }

    std::cout << "squishing AB11" << std::endl;
    printaMatrix(AB11);

    pbtf2(Uplo::Upper, AB11);
    std::cout << "\npbtf2 AB11 done" << std::endl;

    // unsquishing AB11
    for (idx_t j = 0; j < m; j++) {
        for (idx_t i =
                 std::max(0, static_cast<int>(j) - static_cast<int>(m - 1));
             i < j + 1; i++) {
            A11(i, j) = AB11(i + m - 1 - j, j);
        }
    }

    for (idx_t j = 0; j < k; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            AB(i + kd - j, j) = A(i, j);
        }
    }
    // std::vector<T> AB_;
    // auto AB = new_matrix(AB_, kd + 1, n);

    // std::vector<T> AB11_;
    // auto AB11 = new_matrix(AB11_, kd + 1, k);

    // //squishing A00
    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) -
    //     static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         AB(i + kd - j, j) = A00(i, j);
    //     }
    // }
    // // std::cout << "\nAB = " << std::endl;
    // // printaMatrix(AB);

    // pbtf2(Uplo::Upper, AB);

    // //full
    // for (idx_t j = 0; j < n; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) -
    //     static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         A00(i, j) = AB(i + kd - j, j);
    //     }
    // }

    // // std::cout << "C00 = " << std::endl;
    // // printaMatrix(A00);

    // trsm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1),
    // A00, A01);

    // herk(Uplo::Upper, Op::ConjTrans, real_t(-1), A01, real_t(1), A11);

    // for (idx_t j = 0; j < k; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) -
    //     static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         AB11(i + kd - j, j) = A11(i, j);
    //     }
    // }

    // pbtf2(Uplo::Upper, AB11);

    // for (idx_t j = 0; j < k; j++) {
    //     for (idx_t i = std::max(0, static_cast<int>(j) -
    //     static_cast<int>(kd));
    //          i < j + 1; i++) {
    //         A11(i, j) = AB11(i + kd - j, j);
    //     }
    // }
}
template <typename matrix_t>
void pbtrf_cheat_full(matrix_t& A, std::size_t kd)
{
    using T = tlapack::type_t<matrix_t>;
    using idx_t = tlapack::size_type<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using range = pair<idx_t, idx_t>;
    tlapack::Create<matrix_t> new_matrix;

    auto A00 = slice(A, range(0, 13), range(0, 13));
    auto A01 = slice(A, range(0, 13), range(13, 20));
    auto A11 = slice(A, range(13, 20), range(13, 20));

    // std::cout << "\nA00 = " << std::endl;
    // printaMatrix(A00);
    // std::cout << "\nA01 = " << std::endl;
    // printaMatrix(A01);
    // std::cout << "\nA11 = " << std::endl;
    // printaMatrix(A11);

    idx_t n = ncols(A00);
    idx_t k = ncols(A11);

    std::vector<T> AB_;
    auto AB = new_matrix(AB_, kd + 1, n);

    std::vector<T> AB11_;
    auto AB11 = new_matrix(AB11_, kd + 1, k);

    // squishing A00
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            AB(i + kd - j, j) = A00(i, j);
        }
    }
    // std::cout << "\nAB = " << std::endl;
    // printaMatrix(AB);

    pbtf2(Uplo::Upper, AB);

    // full
    for (idx_t j = 0; j < n; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            A00(i, j) = AB(i + kd - j, j);
        }
    }

    // std::cout << "C00 = " << std::endl;
    // printaMatrix(A00);

    trsm(Side::Left, Uplo::Upper, Op::ConjTrans, Diag::NonUnit, real_t(1), A00,
         A01);

    herk(Uplo::Upper, Op::ConjTrans, real_t(-1), A01, real_t(1), A11);

    for (idx_t j = 0; j < k; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            AB11(i + kd - j, j) = A11(i, j);
        }
    }

    pbtf2(Uplo::Upper, AB11);

    for (idx_t j = 0; j < k; j++) {
        for (idx_t i = std::max(0, static_cast<int>(j) - static_cast<int>(kd));
             i < j + 1; i++) {
            A11(i, j) = AB11(i + kd - j, j);
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
    idx_t n = ncols(C);
    real_t beta = 1;
    real_t alpha = -1;

    std::cout << "\nin herkb A11 = " << std::endl;
    printaMatrix(C);
    std::cout << "\nA01 = " << std::endl;
    printaMatrix(A);
    for (idx_t j = 0; j < std::min(static_cast<int>(kd), static_cast<int>(n));
         ++j) {
        std::cout << "\nin loop j" << std::endl;
        for (idx_t i =
                 std::max(static_cast<int>(0), static_cast<int>(k - kd + j));
             i < k; ++i) {
            std::cout << "k - kd + j = " << k << " - " << kd << " + " << j
                      << std::endl;
            TA sum(0);
            std::cout << "in loop i" << std::endl;
            for (idx_t l = i;  // j - kd + n - 1,
                               // std::max(static_cast<int>(0),static_cast<int>(k
                               // - kd - 1 + j))

                 l < k;
                 ++l) {  // std::min(static_cast<int>(kd), static_cast<int>(k))
                std::cout << "in loop l:k - kd -1 + j = " << k << " - " << kd
                          << " - 1 " << " + " << j << std::endl;
                std::cout << "conjA(" << i << ", " << l << ") and A(" << l
                          << ", " << j << ")" << std::endl;

                sum += conj(A(l, i)) * A(l, j);  // j going too far?????
                std::cout << "j = " << j << " i = " << i << " l = " << l
                          << " the sum = " << sum << std::endl;
            }
            std::cout << "C(" << i << ", " << j
                      << ") = " << alpha * sum + beta * C(i, j) << std::endl;
            C(i, j) = alpha * sum + beta * C(i, j);
        }
        std::cout << "out of loop i" << std::endl;
        real_type<TA> sum(0);
        for (idx_t l = std::max(static_cast<int>(j),
                                static_cast<int>(j - kd + n - 1));
             l < std::min(static_cast<int>(kd), static_cast<int>(k)); ++l) {
            std::cout << "in second l loop" << std::endl;
            sum +=
                real(A(l, j)) * real(A(l, j)) + imag(A(l, j)) * imag(A(l, j));
            std::cout << "real A(" << l << ", " << j << ") = " << real(A(l, j))
                      << " imagA(" << l << ", " << j << ") = " << imag(A(l, j))
                      << std::endl;
        }
        C(j, j) = alpha * sum + beta * real(C(j, j));
        std::cout << "C(" << j << ", " << j
                  << ") = " << alpha * sum + beta * real(C(j, j)) << std::endl;
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

    pbtrf_cheat_squish(A, kd);

    // pbtrf_cheat_full(A, kd);

    // const idx_t n = ncols(A);
    // const idx_t n0 = n / 2;

    // auto A00 = slice(A, range(0, n0), range(0, n0));
    // auto A01 = slice(A, range(0, n0), range(n0, n));
    // auto A11 = slice(A, range(n0, n), range(n0, n));
    // std::cout << "\ndone slices" << std::endl;

    // if (kd >= n / 2) {
    //     std::cout << "base case" << std::endl;
    //     std::cout << "slice A00 = " << std::endl;
    //     printaMatrix(A00);
    //     std::cout << "\nslice A01 = " << std::endl;
    //     printaMatrix(A01);
    //     std::cout << "\nslice A11 = " << std::endl;
    //     printaMatrix(A11);
    //     std::cout << std::endl;
    //     potrf(uplo, A00);
    //     // trsm(Side::Left, Uplo::Lower, Op::ConjTrans, Diag::NonUnit,
    //     real_t(1), A00,
    //     // A01);
    //     trsm_b(A00, A01, kd);
    //     std::cout << "\ntrsm A01 = " << std::endl;
    //     printaMatrix(A01);
    //     std::cout << std::endl;
    // //    herk(Uplo::Upper, Op::ConjTrans, real_t(-1), A01, real_t(1), A11);
    //    herk_b(A01, A11, kd);
    //     std::cout << "A11 herk" << std::endl;
    //     printaMatrix(A11);
    //     std::cout << std::endl;

    //     potrf(uplo, A11);
    // }
    // else {
    //     // std::cout << "slice A00 = " << std::endl;
    //     // printaMatrix(A00);

    //     // std::cout << "\nslice A01 = " << std::endl;
    //     // printaMatrix(A01);

    //     // std::cout << "\nslice A11 = " << std::endl;
    //     // printaMatrix(A11);

    //     pbtrf(uplo, A00, kd);
    //     std::cout << "\nrecursion done" << std::endl;

    //     trsm_b(A00, A01, kd);
    //     std::cout << "trsm done" << std::endl;

    //     std::cout << "\nA01 = " << std::endl;
    //     printaMatrix(A01);

    //     herk_b(A01, A11, kd);
    //     std::cout << "done herk" << std::endl;

    //         std::cout << "\nA11 = " << std::endl;
    //     printaMatrix(A11);

    //     pbtrf(uplo, A11, kd);
    // }
    // std::cout << "\nA11 = " << std::endl;
    // printaMatrix(A11);
}
}  // namespace tlapack
   // namespace tlapack

#endif