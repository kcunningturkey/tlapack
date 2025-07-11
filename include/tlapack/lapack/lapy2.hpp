/// @file lapy2.hpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
/// @note Adapted from @see
/// https://github.com/langou/latl/blob/master/include/lapy2.h
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_LAPY2_HH
#define TLAPACK_LAPY2_HH

#include "tlapack/base/utils.hpp"

namespace tlapack {

/** Finds $\sqrt{x^2+y^2}$, taking care not to cause unnecessary overflow.
 *
 * @return $\sqrt{x^2+y^2}$
 *
 * @param[in] x scalar value x
 * @param[in] y scalar value y
 *
 * @ingroup auxiliary
 */
template <TLAPACK_REAL TX,
          TLAPACK_REAL TY,
          enable_if_t<(
                          /* Requires: */
                          is_real<TX> && is_real<TY>),
                      int> = 0>
real_type<TX, TY> lapy2(const TX& x, const TY& y)
{
    // using
    using real_t = real_type<TX, TY>;

    // constants
    const real_t one(1);
    const real_t zero(0);
    const TX xabs = abs(x);
    const TY yabs = abs(y);

    real_t w, z;
    if (xabs > yabs) {
        w = xabs;
        z = yabs;
    }
    else {
        w = yabs;
        z = xabs;
    }

    return (z == zero) ? w : w * sqrt(one + (z / w) * (z / w));
}

}  // namespace tlapack

#endif  // TLAPACK_LAPY2_HH