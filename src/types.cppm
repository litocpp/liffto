export module liffto:types;
export import rstd.core;
export import rstd.alloc;

using namespace rstd::prelude;
using rstd::size_t;

export namespace liffto
{
template<typename T>
concept Float = rstd::mtp::same_as<T, f32> || rstd::mtp::same_as<T, f64>;

template<Float T>
struct Complex {
    T    re {}, im {};
    auto operator==(const Complex&) const -> bool = default;
};
enum class Normalization
{
    None,
    Backward,
    Forward,
    Ortho
};
enum class Backend
{
    Auto,
    Scalar,
    Simd
};
enum class Error
{
    InvalidLength,
    CapacityOverflow,
    AllocationFailed,
    LengthMismatch,
    WorkspaceTooSmall,
    InvalidSpectrum,
    InvalidOption
};

template<Float T, typename A = ::alloc::Global>
class FftPlan;
template<Float T, typename A = ::alloc::Global>
class RealFftPlan;

} // namespace liffto

namespace liffto
{
template<typename T, typename A>
auto buffer(size_t count, A allocator) -> Result<::alloc::vec::Vec<T, A>, Error> {
    auto values   = ::alloc::vec::Vec<T, A>::new_in(rstd::move(allocator));
    auto reserved = values.try_reserve_exact(usize(count));
    if (reserved.is_err())
        return Err(reserved.unwrap_err() == ::alloc::collections::TryReserveError::CapacityOverflow
                       ? Error::CapacityOverflow
                       : Error::AllocationFailed);
    for (size_t i = 0; i < count; ++i) values.push(T {});
    return Ok(rstd::move(values));
}
} // namespace liffto

export namespace liffto
{
/// Private scratch storage; capacity is measured in scalar T elements.
template<Float T, typename A = ::alloc::Global>
class Workspace {
    ::alloc::vec::Vec<T, A> values_;
    explicit Workspace(::alloc::vec::Vec<T, A> values): values_(rstd::move(values)) {}
    template<Float U, typename B>
    friend class FftPlan;

public:
    Workspace(Workspace&&)                           = default;
    auto        operator=(Workspace&&) -> Workspace& = default;
    auto        len() const noexcept -> usize { return values_.len(); }
    static auto make(usize count) -> Result<Workspace, Error> { return make_in(count, A {}); }
    static auto make_in(usize count, A allocator) -> Result<Workspace, Error> {
        auto values = buffer<T>(count.to_primitive(), rstd::move(allocator));
        if (values.is_err()) return Err(values.unwrap_err());
        return Ok(Workspace(rstd::move(values).unwrap()));
    }
};
} // namespace liffto
