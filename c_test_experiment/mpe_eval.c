// mpe_eval.c
#include <stdlib.h>
#include <flint/flint.h>
#include <flint/nmod_poly.h>  // also brings in nmod.h
#include <gmp.h>              // for mp_limb_t

/**
 * Fast multipoint evaluation over Z/modulusZ using FLINT’s
 * subproduct‐tree evaluator for nmod_poly:
 *
 *   nmod_poly_evaluate_nmod_vec_fast(mp_limb_t *ys,
 *                                    const nmod_poly_t poly,
 *                                    const mp_limb_t *xs,
 *                                    slong n)
 *
 * coeffs[0..num_coeffs-1]: polynomial coefficients mod `modulus`
 * points[0..num_points-1]: evaluation x-values
 * out_vals[0..num_points-1] receives f(x) mod `modulus`
 */
void nmod_mpe_fast(
    unsigned long *coeffs, long num_coeffs,
    unsigned long *points, long num_points,
    unsigned long modulus,
    unsigned long *out_vals
) {
    // 1) initialize polynomial f in Z/modulusZ
    nmod_poly_t poly;
    nmod_poly_init(poly, modulus);

    // 2) set coefficients
    for (long i = 0; i < num_coeffs; i++) {
        if (coeffs[i] != 0UL)
            nmod_poly_set_coeff_ui(poly, i, coeffs[i]);
    }

    // 3) pack evaluation points into FLINT limbs
    mp_limb_t *xs = malloc(sizeof(mp_limb_t) * num_points);
    for (long j = 0; j < num_points; j++) {
        xs[j] = (mp_limb_t)(points[j] % modulus);
    }

    // 4) allocate output limb array
    mp_limb_t *ys = malloc(sizeof(mp_limb_t) * num_points);

    // 5) fast subproduct‐tree multipoint eval
    nmod_poly_evaluate_nmod_vec_fast(ys, poly, xs, num_points);

    // 6) copy back to unsigned long
    for (long j = 0; j < num_points; j++) {
        out_vals[j] = (unsigned long) ys[j];
    }

    // 7) cleanup
    free(xs);
    free(ys);
    nmod_poly_clear(poly);
}
