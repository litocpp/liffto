export module liffto:kernels;
import :types;

using namespace rstd::prelude;
using rstd::size_t;

namespace liffto
{
struct Stage {
    size_t half {}, offset {};
};
template<Float T>
inline constexpr size_t lanes = sizeof(T) == 4 ? 4 : 2;

template<Float T>
auto load(const T* p) {
    return rstd::simd::Simd<T, lanes<T>>::from_slice(
        rstd::slice<T>::from_raw_parts(p, usize(lanes<T>)));
}
template<Float T>
void store(T* p, rstd::simd::Simd<T, lanes<T>> v) {
    v.copy_to_slice(rstd::mut_ref<T[]>::from_raw_parts(p, usize(lanes<T>)));
}

template<Float T, bool Vector, bool Inverse>
void             butterfly(T* re, T* im, const T* wr, const T* wi, size_t half) {
#pragma clang fp contract(off)
    size_t j = 0;
    if constexpr (Vector) {
        for (; j + lanes<T> <= half; j += lanes<T>) {
            auto ar = load(re + j), ai = load(im + j), br = load(re + half + j),
                 bi = load(im + half + j);
            auto cr = load(wr + j), ci = load(wi + j);
            if constexpr (Inverse) ci = -ci;
            auto tr = cr * br - ci * bi, ti = cr * bi + ci * br;
            store(re + j, ar + tr);
            store(im + j, ai + ti);
            store(re + half + j, ar - tr);
            store(im + half + j, ai - ti);
        }
    }
    for (; j < half; ++j) {
        T ar = re[j], ai = im[j], br = re[half + j], bi = im[half + j];
        T cr = wr[j], ci = Inverse ? -wi[j] : wi[j];
        T tr = cr * br - ci * bi, ti = cr * bi + ci * br;
        re[j]        = ar + tr;
        im[j]        = ai + ti;
        re[half + j] = ar - tr;
        im[half + j] = ai - ti;
    }
}

template<Float T, bool Vector>
void             multiply(T* re, T* im, const T* cr, const T* ci, size_t count) {
#pragma clang fp contract(off)
    size_t j = 0;
    if constexpr (Vector) {
        for (; j + lanes<T> <= count; j += lanes<T>) {
            auto ar = load(re + j), ai = load(im + j), br = load(cr + j), bi = load(ci + j);
            store(re + j, ar * br - ai * bi);
            store(im + j, ar * bi + ai * br);
        }
    }
    for (; j < count; ++j) {
        T ar = re[j], ai = im[j];
        re[j] = ar * cr[j] - ai * ci[j];
        im[j] = ar * ci[j] + ai * cr[j];
    }
}
} // namespace liffto
