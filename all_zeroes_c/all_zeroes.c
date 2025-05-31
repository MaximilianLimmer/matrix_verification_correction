#include "all_zeroes.h"
#include <flint/flint.h>
#include <flint/fmpz_mod_poly.h>
#include <flint/fmpz_mod.h>
#include <stdlib.h>

int all_zeroes(int *A_flat, int *B_flat, int p_int, int w_int, int t, int n, int l) {
    fmpz_t p, w;
    fmpz_init_set_ui(p, p_int);
    fmpz_init_set_ui(w, w_int);
    fmpz_mod_ctx_t ctx;
    fmpz_mod_ctx_init(ctx, p);

    // Allocate and initialize omega_q, omega_r
    fmpz_t omega_q[t], omega_r[t];
    for (slong i = 0; i < t; i++) {
        fmpz_init(omega_q[i]);
        fmpz_init(omega_r[i]);
        fmpz_pow_ui(omega_q[i], w, i);
        fmpz_pow_ui(omega_r[i], w, i * l);
        fmpz_mod(omega_q[i], omega_q[i], p);
        fmpz_mod(omega_r[i], omega_r[i], p);
    }

    fmpz_t temp;
    fmpz_init(temp);
    int passed = 1;

    for (slong row = 0; row < l; row++) {
        fmpz_mod_poly_t q, r;
        fmpz_mod_poly_init(q, ctx);
        fmpz_mod_poly_init(r, ctx);

        for (slong i = 0; i < l; i++) {
            fmpz_set_si(temp, A_flat[row * l + i]);
            fmpz_mod(temp, temp, p);
            fmpz_mod_poly_set_coeff_fmpz(q, i, temp, ctx);

            fmpz_set_si(temp, B_flat[row * l + i]);  // ✅ correct: access B row[i]
            fmpz_mod(temp, temp, p);
            fmpz_mod_poly_set_coeff_fmpz(r, i, temp, ctx);
}

        for (slong i = 0; i < t; i++) {
            fmpz_t qval, rval;
            fmpz_init(qval); fmpz_init(rval);

            fmpz_mod_poly_evaluate_fmpz(qval, q, omega_q[i], ctx);
            fmpz_mod_poly_evaluate_fmpz(rval, r, omega_r[i], ctx);
            fmpz_mul(qval, qval, rval);
            fmpz_mod(qval, qval, p);

            if (!fmpz_is_zero(qval)) {
                passed = 0;
            }

            fmpz_clear(qval);
            fmpz_clear(rval);
        }

        fmpz_mod_poly_clear(q, ctx);
        fmpz_mod_poly_clear(r, ctx);
    }

    // Cleanup
    for (slong i = 0; i < t; i++) {
        fmpz_clear(omega_q[i]);
        fmpz_clear(omega_r[i]);
    }
    fmpz_clear(temp);
    fmpz_clear(p);
    fmpz_clear(w);
    fmpz_mod_ctx_clear(ctx);

    return passed;
}
