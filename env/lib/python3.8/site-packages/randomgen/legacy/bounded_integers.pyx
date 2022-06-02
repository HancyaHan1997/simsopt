#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True

import numpy as np

cimport numpy as np

from randomgen.legacy.distributions cimport *

np.import_array()

_integers_types = {"bool": (0, 2),
                   "int8": (-2 ** 7, 2 ** 7),
                   "int16": (-2 ** 15, 2 ** 15),
                   "int32": (-2 ** 31, 2 ** 31),
                   "int64": (-2 ** 63, 2 ** 63),
                   "uint8": (0, 2 ** 8),
                   "uint16": (0, 2 ** 16),
                   "uint32": (0, 2 ** 32),
                   "uint64": (0, 2 ** 64)}


cdef object _legacy_rand_uint64(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_uint64(low, high, size, *state, lock)

    Return random np.uint64 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint64 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint64. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint64_t rng, off, out_val
    cdef uint64_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint64)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < 0x0ULL:
        raise ValueError("low is out of bounds for uint64")
    if high > 0xFFFFFFFFFFFFFFFFULL:
        raise ValueError("high is out of bounds for uint64")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint64_t>(high - low)
    off = <uint64_t>(<uint64_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint64_fill(aug_state, off, rng, 1, &out_val)
        return np.uint64(<uint64_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.uint64)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint64_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint64_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_uint32(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_uint32(low, high, size, *state, lock)

    Return random np.uint32 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint32 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint32. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint32_t rng, off, out_val
    cdef uint32_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint32)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < 0x0UL:
        raise ValueError("low is out of bounds for uint32")
    if high > 0XFFFFFFFFUL:
        raise ValueError("high is out of bounds for uint32")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint32_t>(high - low)
    off = <uint32_t>(<uint32_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint32_fill(aug_state, off, rng, 1, &out_val)
        return np.uint32(<uint32_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.uint32)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint32_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint32_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_uint16(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_uint16(low, high, size, *state, lock)

    Return random np.uint16 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint16 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint16. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint16_t rng, off, out_val
    cdef uint16_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint16)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < 0x0UL:
        raise ValueError("low is out of bounds for uint16")
    if high > 0XFFFFUL:
        raise ValueError("high is out of bounds for uint16")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint16_t>(high - low)
    off = <uint16_t>(<uint16_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint16_fill(aug_state, off, rng, 1, &out_val)
        return np.uint16(<uint16_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.uint16)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint16_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint16_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_uint8(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_uint8(low, high, size, *state, lock)

    Return random np.uint8 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint8 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint8. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint8_t rng, off, out_val
    cdef uint8_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint8)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < 0x0UL:
        raise ValueError("low is out of bounds for uint8")
    if high > 0XFFUL:
        raise ValueError("high is out of bounds for uint8")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint8_t>(high - low)
    off = <uint8_t>(<uint8_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint8_fill(aug_state, off, rng, 1, &out_val)
        return np.uint8(<uint8_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.uint8)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint8_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint8_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_bool(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_bool(low, high, size, *state, lock)

    Return random np.bool integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.bool type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.bool
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for bool. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef bool_t rng, off, out_val
    cdef bool_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.bool_)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < 0x0UL:
        raise ValueError("low is out of bounds for bool")
    if high > 0x1UL:
        raise ValueError("high is out of bounds for bool")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <bool_t>(high - low)
    off = <bool_t>(<bool_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_bool_fill(aug_state, off, rng, 1, &out_val)
        return np.bool_(<bool_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.bool_)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <bool_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_bool_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_int64(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_int64(low, high, size, *state, lock)

    Return random np.int64 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int64 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint64. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint64_t rng, off, out_val
    cdef uint64_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int64)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < -0x8000000000000000LL:
        raise ValueError("low is out of bounds for int64")
    if high > 0x7FFFFFFFFFFFFFFFL:
        raise ValueError("high is out of bounds for int64")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint64_t>(high - low)
    off = <uint64_t>(<int64_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint64_fill(aug_state, off, rng, 1, &out_val)
        return np.int64(<int64_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.int64)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint64_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint64_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_int32(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_int32(low, high, size, *state, lock)

    Return random np.int32 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int32 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint32. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint32_t rng, off, out_val
    cdef uint32_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int32)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < -0x80000000L:
        raise ValueError("low is out of bounds for int32")
    if high > 0x7FFFFFFFL:
        raise ValueError("high is out of bounds for int32")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint32_t>(high - low)
    off = <uint32_t>(<int32_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint32_fill(aug_state, off, rng, 1, &out_val)
        return np.int32(<int32_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.int32)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint32_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint32_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_int16(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_int16(low, high, size, *state, lock)

    Return random np.int16 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int16 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint16. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint16_t rng, off, out_val
    cdef uint16_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int16)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < -0x8000L:
        raise ValueError("low is out of bounds for int16")
    if high > 0x7FFFL:
        raise ValueError("high is out of bounds for int16")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint16_t>(high - low)
    off = <uint16_t>(<int16_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint16_fill(aug_state, off, rng, 1, &out_val)
        return np.int16(<int16_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.int16)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint16_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint16_fill(aug_state, off, rng, cnt, out_data)
        return out_arr

cdef object _legacy_rand_int8(object low, object high, object size,
                            aug_bitgen_t *aug_state, object lock):
    """
    _legacy_rand_int8(low, high, size, *state, lock)

    Return random np.int8 integers from `low` (inclusive) to `high` (exclusive).

    Return random integers from the "discrete uniform" distribution in the
    interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int8 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, one above the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : bit generator
        Bit generator state to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.

    Notes
    -----
    The internal integer generator produces values from the closed
    interval [low, high].  This requires some care since
    high can be out-of-range for uint8. The scalar path leaves
    integers as Python integers until the 1 has been subtracted to
    avoid needing to cast to a larger type.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint8_t rng, off, out_val
    cdef uint8_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int8)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    scalar = ((low_ndim == 0 or (low_ndim == 1 and low_arr.size == 1 and size is not None)) and
              (high_ndim == 0 or (high_ndim == 1 and high_arr.size == 1 and size is not None)))
    if not scalar:
        raise ValueError("Only scalar-compatible inputs are accepted for low and high")

    low = int(low_arr)
    high = int(high_arr)
    # Subtract 1 since internal generator produces on closed interval [low, high]
    high -= 1

    if low < -0x80L:
        raise ValueError("low is out of bounds for int8")
    if high > 0x7FL:
        raise ValueError("high is out of bounds for int8")
    if low > high:  # -1 already subtracted, closed interval
        raise ValueError("low >= high")

    rng = <uint8_t>(high - low)
    off = <uint8_t>(<int8_t>low)
    if size is None:
        with lock:
            legacy_random_bounded_uint8_fill(aug_state, off, rng, 1, &out_val)
        return np.int8(<int8_t>out_val)
    else:
        out_arr = <np.ndarray>np.empty(size, np.int8)
        cnt = np.PyArray_SIZE(out_arr)
        out_data = <uint8_t *>np.PyArray_DATA(out_arr)
        with lock, nogil:
            legacy_random_bounded_uint8_fill(aug_state, off, rng, cnt, out_data)
        return out_arr
