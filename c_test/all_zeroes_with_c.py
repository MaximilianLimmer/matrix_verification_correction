import ctypes
import numpy as np

lib = ctypes.CDLL("./all_zeroes_c/liball_zeroes.so")
lib.all_zeroes.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.all_zeroes.restype = ctypes.c_int

def all_zeroes(A, B, p, w, t, n, l):
    A_np = np.ascontiguousarray(A.reshape(-1).astype(np.int32))
    B_np = np.ascontiguousarray(B.reshape(-1).astype(np.int32))

    A_ptr = A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    B_ptr = B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

    return bool(lib.all_zeroes(A_ptr, B_ptr, int(p), int(w), int(t), int(n), int(l)))