import ctypes, os
import numpy as np

_lib = ctypes.CDLL(os.path.abspath("libmpe.so"))

_sig = [
    ctypes.POINTER(ctypes.c_ulong), ctypes.c_long,  # coeffs, num_coeffs
    ctypes.POINTER(ctypes.c_ulong), ctypes.c_long,  # points, num_points
    ctypes.c_ulong,                                 # modulus
    ctypes.POINTER(ctypes.c_ulong)                  # out_vals
]

_lib.nmod_mpe_fast.argtypes = _sig
_lib.nmod_mpe_fast.restype  = None

def nmod_mpe_fast(coeffs, points, modulus):
    # bring everything into the [0..modulus) range first
    a = np.asarray(coeffs, dtype=np.int64) % modulus
    x = np.asarray(points, dtype=np.int64) % modulus

    # now cast into uint64 without error
    u_coeffs = a.astype(np.uint64)
    u_points = x.astype(np.uint64)
    out = np.zeros(len(u_points), dtype=np.uint64)

    _lib.nmod_mpe_fast(
        u_coeffs.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),
        len(u_coeffs),
        u_points.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),
        len(u_points),
        ctypes.c_ulong(modulus),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ulong)),
    )
    return out.tolist()