export module liffto:plan;
export import :types;
import :kernels;

using namespace rstd::prelude;
using rstd::size_t;

export namespace liffto
{
/// Immutable transform configuration. Execution reads the complete input before
/// writing output.
template<Float T, typename A>
class FftPlan {
    static_assert(rstd::mtp::copy<A>, "plan allocator must be copy constructible");
    size_t                           n_, m_, twiddles_, chirp_, spectrum_, stage_count_ {};
    Backend                          backend_;
    T                                forward_scale_, inverse_scale_;
    array<Stage, sizeof(size_t) * 8> stages_ {};
    ::alloc::vec::Vec<T, A>          coefficients_;
    ::alloc::vec::Vec<usize, A>      permutation_;
    FftPlan(size_t                      n,
            size_t                      m,
            Backend                     backend,
            T                           fs,
            T                           is,
            ::alloc::vec::Vec<T, A>     data,
            ::alloc::vec::Vec<usize, A> permutation)
        : n_(n),
          m_(m),
          twiddles_(m - 1),
          chirp_(2 * (m - 1)),
          spectrum_(chirp_ + (m == n ? 0 : 2 * n)),
          backend_(backend),
          forward_scale_(fs),
          inverse_scale_(is),
          coefficients_(rstd::move(data)),
          permutation_(rstd::move(permutation)) {}
    template<Float U, typename B>
    friend class RealFftPlan;

    template<bool Vector, bool Inverse = false>
    void radix(T* re, T* im) const {
        const auto* permutation = permutation_.as_slice().as_raw_ptr();
        for (size_t i = 0; i < m_; ++i) {
            auto j = permutation[i].to_primitive();
            if (i < j) {
                T r = re[i], v = im[i];
                re[i] = re[j];
                im[i] = im[j];
                re[j] = r;
                im[j] = v;
            }
        }
        const auto* data = coefficients_.as_slice().as_raw_ptr();
        for (size_t s = 0; s < stage_count_; ++s) {
            const auto& stage = stages_[usize(s)];
            for (size_t base = 0; base < m_; base += 2 * stage.half)
                butterfly<T, Vector, Inverse>(re + base,
                                              im + base,
                                              data + stage.offset,
                                              data + twiddles_ + stage.offset,
                                              stage.half);
        }
    }
    template<bool Vector>
    void transform(T* re, T* im) const {
        const auto* data = coefficients_.as_slice().as_raw_ptr();
        if (m_ != n_) multiply<T, Vector>(re, im, data + chirp_, data + chirp_ + n_, n_);
        radix<Vector>(re, im);
        if (m_ != n_) {
            multiply<T, Vector>(re, im, data + spectrum_, data + spectrum_ + m_, m_);
            radix<Vector, true>(re, im);
            multiply<T, Vector>(re, im, data + chirp_, data + chirp_ + n_, n_);
        }
    }
    template<typename Read, typename Write, typename W>
    auto execute(Read read, Write write, Workspace<T, W>& workspace, bool inverse) const
        -> Result<empty, Error> {
#pragma clang fp contract(off)
        if (workspace.len() < workspace_len()) return Err(Error::WorkspaceTooSmall);
        auto* re = workspace.values_.as_mut_slice().as_raw_ptr();
        auto* im = re + m_;
        for (size_t i = 0; i < n_; ++i) {
            auto v = read(i);
            re[i]  = v.re;
            im[i]  = inverse ? -v.im : v.im;
        }
        for (size_t i = n_; i < m_; ++i) {
            re[i] = T {};
            im[i] = T {};
        }
        if (backend_ == Backend::Simd)
            transform<true>(re, im);
        else
            transform<false>(re, im);
        T scale = inverse ? inverse_scale_ : forward_scale_;
        if (m_ != n_) scale = scale / T(m_);
        for (size_t i = 0; i < n_; ++i)
            write(i, Complex<T> { re[i] * scale, (inverse ? -im[i] : im[i]) * scale });
        return Ok(empty {});
    }
    template<typename W>
    auto complex_execute(rstd::slice<Complex<T>>     input,
                         rstd::mut_ref<Complex<T>[]> output,
                         Workspace<T, W>&            workspace,
                         bool                        inverse) const -> Result<empty, Error> {
        if (input.len() != len() || output.len() != len()) return Err(Error::LengthMismatch);
        return execute(
            [&](size_t i) {
                return input[usize(i)];
            },
            [&](size_t i, Complex<T> v) {
                output[usize(i)] = v;
            },
            workspace,
            inverse);
    }

public:
    FftPlan(FftPlan&&)                    = default;
    auto operator=(FftPlan&&) -> FftPlan& = default;
    auto len() const noexcept -> usize { return usize(n_); }
    auto workspace_len() const noexcept -> usize { return usize(2 * m_); }
    auto backend() const noexcept -> Backend { return backend_; }
    auto convolution_len() const noexcept -> usize { return usize(m_); }
    auto table_bytes() const noexcept -> usize {
        return coefficients_.len() * usize(sizeof(T)) + permutation_.len() * usize(sizeof(usize));
    }
    static auto make(usize         n,
                     Normalization normalization = Normalization::Backward,
                     Backend       backend       = Backend::Auto) -> Result<FftPlan, Error> {
        return make_in(n, A {}, normalization, backend);
    }
    static auto make_in(usize         length,
                        A             allocator,
                        Normalization normalization = Normalization::Backward,
                        Backend       backend       = Backend::Auto) -> Result<FftPlan, Error> {
        auto n = length.to_primitive();
        if (! n) return Err(Error::InvalidLength);
        if (normalization != Normalization::None && normalization != Normalization::Backward &&
            normalization != Normalization::Forward && normalization != Normalization::Ortho)
            return Err(Error::InvalidOption);
        if (backend != Backend::Auto && backend != Backend::Scalar && backend != Backend::Simd)
            return Err(Error::InvalidOption);
        // This also bounds 2*N, power-of-two rounding, and every byte-size
        // calculation below.
        auto limit = usize::MAX.to_primitive() / (16 * sizeof(T) + 2 * sizeof(usize));
        if (n > limit) return Err(Error::CapacityOverflow);
        size_t m = n;
        if (n & (n - 1)) {
            m = 1;
            while (m < 2 * n - 1) m *= 2;
        }
        if (m > limit) return Err(Error::CapacityOverflow);
        auto count = 2 * (m - 1) + (m == n ? 0 : 2 * n + 2 * m);
        auto data  = buffer<T>(count, allocator);
        if (data.is_err()) return Err(data.unwrap_err());
        auto permutation = buffer<usize>(m, allocator);
        if (permutation.is_err()) return Err(permutation.unwrap_err());
        T fs(1), is(1);
        if (normalization == Normalization::Backward) is = T(1) / T(n);
        if (normalization == Normalization::Forward) fs = T(1) / T(n);
        if (normalization == Normalization::Ortho) fs = is = T(1) / T(n).sqrt();
        if (backend == Backend::Auto) backend = m >= 16 ? Backend::Simd : Backend::Scalar;
        FftPlan plan(
            n, m, backend, fs, is, rstd::move(data).unwrap(), rstd::move(permutation).unwrap());
        auto*  table  = plan.coefficients_.as_mut_slice().as_raw_ptr();
        size_t offset = 0;
        for (size_t half = 1; half < m; half *= 2) {
            plan.stages_[usize(plan.stage_count_++)] = Stage { half, offset };
            for (size_t j = 0; j < half; ++j) {
                f64 angle                          = -f64::consts::TAU * f64(j) / f64(2 * half);
                table[offset + j]                  = rstd::as_cast<T>(angle.cos());
                table[plan.twiddles_ + offset + j] = rstd::as_cast<T>(angle.sin());
            }
            offset += half;
        }
        size_t rev = 0;
        for (size_t i = 0; i < m; ++i) {
            plan.permutation_[usize(i)] = usize(rev);
            size_t bit                  = m / 2;
            while (bit && (rev & bit)) {
                rev ^= bit;
                bit >>= 1;
            }
            rev ^= bit;
        }
        if (m != n) {
            // j^2 mod 2N via bounded modular increments, without ever forming j^2.
            size_t phase = 0, modulus = 2 * n;
            for (size_t j = 0; j < n; ++j) {
                f64 angle = -f64::consts::PI * f64(phase) / f64(n);
                T   cr = rstd::as_cast<T>(angle.cos()), ci = rstd::as_cast<T>(angle.sin());
                table[plan.chirp_ + j]        = cr;
                table[plan.chirp_ + n + j]    = ci;
                table[plan.spectrum_ + j]     = cr;
                table[plan.spectrum_ + m + j] = -ci;
                if (j) {
                    table[plan.spectrum_ + m - j]     = cr;
                    table[plan.spectrum_ + 2 * m - j] = -ci;
                }
                size_t delta = 2 * j + 1;
                phase        = phase >= modulus - delta ? phase - (modulus - delta) : phase + delta;
            }
            // The kernel spectrum occupies a disjoint section of the coefficient
            // allocation.
            plan.radix<false>(table + plan.spectrum_, table + plan.spectrum_ + m);
        }
        return Ok(rstd::move(plan));
    }
    template<typename W>
    auto forward(rstd::slice<Complex<T>>     input,
                 rstd::mut_ref<Complex<T>[]> output,
                 Workspace<T, W>&            workspace) const -> Result<empty, Error> {
        return complex_execute(input, output, workspace, false);
    }
    template<typename W>
    auto inverse(rstd::slice<Complex<T>>     input,
                 rstd::mut_ref<Complex<T>[]> output,
                 Workspace<T, W>&            workspace) const -> Result<empty, Error> {
        return complex_execute(input, output, workspace, true);
    }
    template<typename W>
    auto forward_in_place(rstd::mut_ref<Complex<T>[]> values, Workspace<T, W>& workspace) const
        -> Result<empty, Error> {
        return forward(rstd::slice<Complex<T>>::from_raw_parts(values.as_raw_ptr(), values.len()),
                       values,
                       workspace);
    }
    template<typename W>
    auto inverse_in_place(rstd::mut_ref<Complex<T>[]> values, Workspace<T, W>& workspace) const
        -> Result<empty, Error> {
        return inverse(rstd::slice<Complex<T>>::from_raw_parts(values.as_raw_ptr(), values.len()),
                       values,
                       workspace);
    }
};
} // namespace liffto
