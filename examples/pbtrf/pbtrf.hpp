#ifndef TLAPACK_PBTRF_HH
#define TLAPACK_PBTRF_HH

#include "tlapack/base/utils.hpp"
#include "tlapack/blas/herk.hpp"
#include "tlapack/blas/trsm.hpp"
#include "tlapack/lapack/pbtf0.hpp"

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

struct BlockedBandedCholeskyOpts : public EcOpts {
    constexpr BlockedBandedCholeskyOpts(const EcOpts& opts = {})
        : EcOpts(opts) {};

    size_t nb = 32;  ///< Block size
};

template <TLAPACK_UPLO uplo_t, TLAPACK_SMATRIX matrix_t>
void pbtrf(uplo_t uplo,
           matrix_t& AB,
           const BlockedBandedCholeskyOpts& opts = {})
{
    using T = tlapack::type_t<matrix_t>;
    using real_t = tlapack::real_type<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    Create<matrix_t> new_matrix;

    idx_t n = ncols(AB);
    idx_t kd = nrows(AB) - 1;
    const idx_t nb = opts.nb;

    if (nb < 1 || nb > kd) {
        pbtf0(uplo, AB);
    }
    else {
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
            for (idx_t i = 0; i < n; i += nb) {
                // idx_t ib = std::min(static_cast<int>(nb), static_cast<int>(n
                // - i));
                idx_t ib;
                if (n < nb + i) {
                    ib = n - i;
                }
                else  //(nb + i < n)
                {
                    ib = nb;
                }

                auto AB00 =
                    slice(AB, range(kd - ib + 1, kd + 1), range(i, i + ib));

                AB00.ptr = &AB00.ptr[ib - 1];
                AB00.ldim -= 1;

                potf2(tlapack::Uplo::Upper, AB00);

                if (i + ib < n) {
                    // idx_t i2 = std::min(static_cast<int>(kd-ib),
                    // static_cast<int>(n - i - ib));
                    idx_t i2;
                    if (kd + i < n) {
                        i2 = kd - ib;
                    }
                    else {
                        i2 = n - i - ib;
                    }

                    if (i2 > 0) {

                        auto AB01 = slice(
                            AB, range(kd - ib, kd),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));
                        AB01.ldim -= 1;

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, AB01);

                        auto AB11 = slice(
                            AB, range(kd + 1 - i2, kd + 1),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));

                        AB11.ptr = &AB11.ptr[i2 - 1];
                        AB11.ldim -= 1;

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), AB01, real_t(1), AB11);

                    }

                    // int i3 = std::min(static_cast<int>(ib),
                    // static_cast<int>(n- i
                    // - kd)); // change to int i hate unsigned ints omg
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
                        auto work02 = slice(work, range(0, ib), range(0, i3));

                        for (idx_t jj = 0; jj < i3; jj++) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                work02(ii, jj) = AB(ii - jj, jj + i + kd);
                            }
                        }

                        trsm(tlapack::Side::Left, tlapack::Uplo::Upper,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, work02);

                        auto AB12 =
                            slice(AB, range(kd - i2, kd),
                                  range(i + kd,
                                        std::min(static_cast<int>(i + kd + i3),
                                                 static_cast<int>(n))));
                        AB12.ldim -= 1;

                        auto AB01 = slice(
                            AB, range(kd - ib, kd),
                            range(i + ib, std::min(static_cast<int>(i + kd),
                                                   static_cast<int>(n))));
                        AB01.ldim -= 1;

                        gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                             real_t(-1), AB01, work02, real_t(1), AB12);

                        auto AB22 = slice(
                            AB, range(kd - i3 + 1, kd + 1),
                            range(i + ib + i2,
                                  std::min(static_cast<int>(i + 2 * ib + i2),
                                           static_cast<int>(n))));
                        AB22.ptr = &AB22.ptr[i3 - 1];
                        AB22.ldim -= 1;

                        herk(tlapack::Uplo::Upper, tlapack::Op::ConjTrans,
                             real_t(-1), work02, real_t(1), AB22);

                        for (idx_t jj = 0; jj < i3; ++jj) {
                            for (idx_t ii = jj; ii < ib; ++ii) {
                                AB(ii - jj, jj + i + kd) = work02(ii, jj);
                            }
                        }
                    }
                }
            }

        }
        else {
            for (idx_t i = 0; i < n; i += nb)
            // for (idx_t i = 0; i <= 0; i++)
            {
                idx_t ib;
                if (nb + i < n) {
                    ib = nb;
                }
                else {
                    ib = n - i;
                }

                auto AB00 = slice(AB, range(0, ib), range(i, ib + i));
                AB00.ldim -= 1;

                potf2(tlapack::Uplo::Lower, AB00);

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
                        auto AB10 =
                            slice(AB, range(ib, ib + i2), range(i, ib + i));
                        AB10.ldim -= 1;

                        trsm(tlapack::Side::Right, tlapack::Uplo::Lower,
                             tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                             real_t(1), AB00, AB10);

                        auto AB11 =
                            slice(AB, range(0, i2), range(i + ib, i + ib + i2));
                        AB11.ldim -= 1;

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), AB10,
                             real_t(1), AB11);
                    }

                    if (i3 > 0) {
                        auto AB10 =
                            slice(AB, range(ib, ib + i2), range(i, ib + i));
                        AB10.ldim -= 1;

                        auto AB20 = slice(AB, range(kd - i3 + 1, kd + 1),
                                          range(i, ib + i));
                        AB20.ptr = &AB20.ptr[i3 - 1];
                        AB20.ldim -= 1;

                        auto work20 = slice(work, range(0, i3), range(0, ib));

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < std::min(static_cast<int>(jj + 1),
                                               static_cast<int>(i3));
                                 ++ii) {
                                work20(ii, jj) = AB(kd - jj + ii, jj + i);
                            }
                        }

                        trsm(tlapack::Side::Right, uplo, tlapack::Op::ConjTrans,
                             tlapack::Diag::NonUnit, real_t(1), AB00, work20);

                        auto AB21 = slice(AB, range(i2, i2 + i3),
                                          range(i + ib, i + ib + i2));
                        AB21.ldim -= 1;

                        gemm(tlapack::Op::NoTrans, tlapack::Op::ConjTrans,
                             real_t(-1), work20, AB10, real_t(1), AB21);

                        auto AB22 =
                            slice(AB, range(0, i3),
                                  range(i + ib + i2,
                                        i + ib + i2 + std::min((ib), (i3))));
                        AB22.ldim -= 1;

                        herk(uplo, tlapack::Op::NoTrans, real_t(-1), work20,
                             real_t(1), AB22);

                        for (idx_t jj = 0; jj < ib; jj++) {
                            for (idx_t ii = 0;
                                 ii < std::min(static_cast<int>(jj + 1),
                                               static_cast<int>(i3));
                                 ++ii) {
                                AB(kd - jj + ii, jj + i) = work20(ii, jj);
                            }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace tlapack
   // namespace tlapack

#endif