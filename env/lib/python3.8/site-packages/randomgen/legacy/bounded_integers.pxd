from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    intptr_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)

import numpy as np

cimport numpy as np

ctypedef np.npy_bool bool_t

from randomgen.legacy.distributions cimport aug_bitgen_t


cdef inline uint64_t _gen_mask(uint64_t max_val) nogil:
    """Mask generator for use in bounded random numbers"""
    # Smallest bit mask >= max
    cdef uint64_t mask = max_val
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    mask |= mask >> 32
    return mask

cdef object _legacy_rand_uint64(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_uint32(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_uint16(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_uint8(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_bool(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_int64(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_int32(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_int16(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
cdef object _legacy_rand_int8(object low, object high, object size, aug_bitgen_t *aug_state, object lock)
