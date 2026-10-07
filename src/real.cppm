export module liffto:real;
export import :plan;

using namespace rstd::prelude;
using rstd::size_t;

export namespace liffto
{
/// A real transform with an explicit original length, including odd lengths.
template<Float T, typename A>
class RealFftPlan {
    FftPlan<T, A> plan_;
    explicit RealFftPlan(FftPlan<T, A> plan): plan_(rstd::move(plan)) {}

public:
    RealFftPlan(RealFftPlan&&)                           = default;
    auto        operator=(RealFftPlan&&) -> RealFftPlan& = default;
    auto        len() const noexcept -> usize { return plan_.len(); }
    auto        spectrum_len() const noexcept -> usize { return len() / usize(2) + usize(1); }
    auto        workspace_len() const noexcept -> usize { return plan_.workspace_len(); }
    auto        backend() const noexcept -> Backend { return plan_.backend(); }
    static auto make(usize         n,
                     Normalization normalization = Normalization::Backward,
                     Backend       backend       = Backend::Auto) -> Result<RealFftPlan, Error> {
        return make_in(n, A {}, normalization, backend);
    }
    static auto make_in(usize         n,
                        A             allocator,
                        Normalization normalization = Normalization::Backward,
                        Backend       backend       = Backend::Auto) -> Result<RealFftPlan, Error> {
        auto plan = FftPlan<T, A>::make_in(n, rstd::move(allocator), normalization, backend);
        if (plan.is_err()) return Err(plan.unwrap_err());
        return Ok(RealFftPlan(rstd::move(plan).unwrap()));
    }
    template<typename W>
    auto forward(rstd::slice<T>              input,
                 rstd::mut_ref<Complex<T>[]> output,
                 Workspace<T, W>&            workspace) const -> Result<empty, Error> {
        if (input.len() != len() || output.len() != spectrum_len())
            return Err(Error::LengthMismatch);
        auto n = len().to_primitive();
        return plan_.execute(
            [&](size_t i) {
                return Complex<T> { input[usize(i)], T {} };
            },
            [&](size_t i, Complex<T> v) {
                if (i <= n / 2) {
                    if (i == 0 || (! (n & 1) && i == n / 2)) v.im = T {};
                    output[usize(i)] = v;
                }
            },
            workspace,
            false);
    }
    template<typename W>
    auto inverse(rstd::slice<Complex<T>> input,
                 rstd::mut_ref<T[]>      output,
                 Workspace<T, W>&        workspace) const -> Result<empty, Error> {
        if (input.len() != spectrum_len() || output.len() != len())
            return Err(Error::LengthMismatch);
        auto n = len().to_primitive();
        if (input[usize()].im != T {} || (! (n & 1) && input[usize(n / 2)].im != T {}))
            return Err(Error::InvalidSpectrum);
        return plan_.execute(
            [&](size_t i) {
                if (i <= n / 2) return input[usize(i)];
                auto v = input[usize(n - i)];
                return Complex<T> { v.re, -v.im };
            },
            [&](size_t i, Complex<T> v) {
                output[usize(i)] = v.re;
            },
            workspace,
            true);
    }
};
} // namespace liffto
