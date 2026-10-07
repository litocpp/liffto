#include <rstd/test/gtest.hpp>
import liffto;
import rstd;
using namespace rstd::prelude;
using namespace liffto;
using rstd::size_t;

TEST(FftMemory, InvalidArgumentsPreserveOutput) {
    EXPECT_EQ(FftPlan<f32>::make(usize()).unwrap_err(), Error::InvalidLength);
    EXPECT_EQ(FftPlan<f64>::make(usize::MAX).unwrap_err(), Error::CapacityOverflow);
    EXPECT_EQ(FftPlan<f64>::make(usize::MAX / usize(32)).unwrap_err(), Error::CapacityOverflow);
    EXPECT_EQ(Workspace<f64>::make(usize::MAX).unwrap_err(), Error::CapacityOverflow);
    EXPECT_EQ(FftPlan<f64>::make(usize(4), static_cast<Normalization>(99)).unwrap_err(),
              Error::InvalidOption);
    auto                   plan    = FftPlan<f32>::make(usize(8)).unwrap();
    auto                   tiny    = Workspace<f32>::make(usize(15)).unwrap();
    auto                   scratch = Workspace<f32>::make(plan.workspace_len()).unwrap();
    array<Complex<f32>, 8> values {};
    for (auto& v : values) v = { f32(123), f32(456) };
    EXPECT_EQ(plan.forward_in_place(values.as_mut_slice(), tiny).unwrap_err(),
              Error::WorkspaceTooSmall);
    auto short_input =
        rstd::slice<Complex<f32>>::from_raw_parts(values.as_mut_slice().as_raw_ptr(), usize(7));
    EXPECT_EQ(plan.forward(short_input, values.as_mut_slice(), scratch).unwrap_err(),
              Error::LengthMismatch);
    auto short_output =
        rstd::mut_ref<Complex<f32>[]>::from_raw_parts(values.as_mut_slice().as_raw_ptr(), usize(7));
    EXPECT_EQ(plan.inverse(values.as_slice(), short_output, scratch).unwrap_err(),
              Error::LengthMismatch);
    for (auto v : values) {
        EXPECT_EQ(v.re, f32(123));
        EXPECT_EQ(v.im, f32(456));
    }
}
TEST(RealFft, InvalidSpectrumPreservesOutput) {
    auto                   plan    = RealFftPlan<f64>::make(usize(8)).unwrap();
    auto                   scratch = Workspace<f64>::make(plan.workspace_len()).unwrap();
    array<Complex<f64>, 5> spectrum {};
    array<f64, 8>          output {};
    for (auto& v : output) v = f64(42);
    for (usize index : array<decltype(usize()), 2> { usize(), usize(4) })
        for (f64 bad : array<decltype(f64(1)), 2> { f64(1), f64(__builtin_nan("")) }) {
            spectrum[index].im = bad;
            EXPECT_EQ(
                plan.inverse(spectrum.as_slice(), output.as_mut_slice(), scratch).unwrap_err(),
                Error::InvalidSpectrum);
            spectrum[index].im = f64();
        }
    auto short_input =
        rstd::slice<Complex<f64>>::from_raw_parts(spectrum.as_mut_slice().as_raw_ptr(), usize(4));
    EXPECT_EQ(plan.inverse(short_input, output.as_mut_slice(), scratch).unwrap_err(),
              Error::LengthMismatch);
    auto tiny = Workspace<f64>::make(usize()).unwrap();
    EXPECT_EQ(plan.inverse(spectrum.as_slice(), output.as_mut_slice(), tiny).unwrap_err(),
              Error::WorkspaceTooSmall);
    for (auto v : output) EXPECT_EQ(v, f64(42));
}

template<typename T>
void aliases() {
    for (size_t n : array<decltype(size_t(1)), 7> {
             size_t(1), size_t(3), size_t(8), size_t(15), size_t(16), size_t(17), size_t(33) }) {
        auto plan    = FftPlan<T>::make(usize(n), Normalization::Backward, Backend::Simd).unwrap();
        auto scratch = Workspace<T>::make(plan.workspace_len() + usize(9)).unwrap();
        array<Complex<T>, 80> storage {};
        array<Complex<T>, 33> original {}, expected {};
        for (size_t i = 0; i < n; ++i) original[usize(i)] = { T(i) / T(3), T(-2) * T(i) };
        auto input =
            rstd::slice<Complex<T>>::from_raw_parts(original.as_mut_slice().as_raw_ptr(), usize(n));
        auto reference = rstd::mut_ref<Complex<T>[]>::from_raw_parts(
            expected.as_mut_slice().as_raw_ptr(), usize(n));
        plan.forward(input, reference, scratch).unwrap();
        for (size_t output_offset :
             array<decltype(size_t(1)), 4> { size_t(1), size_t(2), size_t(3), size_t(40) }) {
            for (auto& v : storage) v = { T(999), T(-999) };
            for (size_t i = 0; i < n; ++i) storage[usize(i + 2)] = original[usize(i)];
            auto source = rstd::slice<Complex<T>>::from_raw_parts(
                storage.as_mut_slice().as_raw_ptr() + 2, usize(n));
            auto output = rstd::mut_ref<Complex<T>[]>::from_raw_parts(
                storage.as_mut_slice().as_raw_ptr() + output_offset, usize(n));
            array<Complex<T>, 80> before = storage;
            ASSERT_TRUE(plan.forward(source, output, scratch).is_ok());
            for (size_t i = 0; i < 80; ++i) {
                if (i >= output_offset && i < output_offset + n)
                    EXPECT_EQ(storage[usize(i)], expected[usize(i - output_offset)]);
                else
                    EXPECT_EQ(storage[usize(i)], before[usize(i)]);
            }
        }
    }
}
TEST(FftMemory, AliasingOffsetsAndSentinels) {
    aliases<f32>();
    aliases<f64>();
}

struct FailState {
    unsigned calls {}, fail_at {};
};
struct FailingAllocator {
    FailState*      state;
    ::alloc::Global upstream;
    auto            allocate(rstd::alloc::Layout layout) const
        -> Result<rstd::alloc::Allocation, rstd::alloc::AllocError> {
        if (++state->calls == state->fail_at) return Err(rstd::alloc::AllocError {});
        return rstd::as<rstd::alloc::Allocator>(upstream).allocate(layout);
    }
    void deallocate(void* pointer, rstd::alloc::Layout layout) const noexcept {
        rstd::as<rstd::alloc::Allocator>(upstream).deallocate(pointer, layout);
    }
};
namespace rstd
{
template<>
struct Impl<alloc::Allocator, FailingAllocator>
    : DefaultInImpl<alloc::Allocator, FailingAllocator> {
    auto allocate(alloc::Layout layout) const -> Result<alloc::Allocation, alloc::AllocError> {
        return this->self().allocate(layout);
    }
    void deallocate(void* p, alloc::Layout layout) const noexcept {
        this->self().deallocate(p, layout);
    }
};
} // namespace rstd
TEST(FftMemory, AllocationFailureReleasesPartialPlan) {
    using A = rstd::alloc::TrackingAllocator<FailingAllocator>;
    for (unsigned failure : array<decltype(1U), 2> { 1U, 2U }) {
        FailState                      state { 0, failure };
        rstd::alloc::AllocationTracker tracker;
        auto plan = FftPlan<f64, A>::make_in(usize(17), A(FailingAllocator { &state }, tracker));
        EXPECT_EQ(plan.unwrap_err(), Error::AllocationFailed);
        EXPECT_EQ(tracker.snapshot().live_bytes, u64());
    }
    FailState                      state { 0, 1 };
    rstd::alloc::AllocationTracker tracker;
    EXPECT_EQ((Workspace<f32, A>::make_in(usize(64), A(FailingAllocator { &state }, tracker))
                   .unwrap_err()),
              Error::AllocationFailed);
    EXPECT_EQ(tracker.snapshot().live_bytes, u64());
}
TEST(FftMemory, NoExecuteAllocationsAndMoveOwnership) {
    using A = rstd::alloc::TrackingAllocator<>;
    rstd::alloc::AllocationTracker tracker;
    {
        A    allocator(::alloc::Global {}, tracker);
        auto original  = FftPlan<f64, A>::make_in(usize(17), allocator).unwrap();
        auto plan      = FftPlan<f64, A>::make_in(usize(8), allocator).unwrap();
        plan           = rstd::move(original);
        auto real      = RealFftPlan<f64, A>::make_in(usize(16), allocator).unwrap();
        auto workspace = Workspace<f64, A>::make_in(plan.workspace_len(), allocator).unwrap();
        auto moved     = Workspace<f64, A>::make_in(usize(3), allocator).unwrap();
        moved          = rstd::move(workspace);
        array<Complex<f64>, 17> values {};
        values[usize()].re = f64(1);
        array<f64, 16>         input {};
        array<Complex<f64>, 9> half {};
        auto                   before = tracker.snapshot();
        for (int i = 0; i < 20; ++i) {
            plan.forward_in_place(values.as_mut_slice(), moved).unwrap();
            plan.inverse_in_place(values.as_mut_slice(), moved).unwrap();
            real.forward(input.as_slice(), half.as_mut_slice(), moved).unwrap();
            real.inverse(half.as_slice(), input.as_mut_slice(), moved).unwrap();
        }
        auto after = tracker.snapshot();
        EXPECT_EQ(before.allocations, after.allocations);
        EXPECT_EQ(before.grows, after.grows);
        EXPECT_EQ(before.live_bytes, after.live_bytes);
        EXPECT_EQ(before.deallocations, after.deallocations);
    }
    EXPECT_EQ(tracker.snapshot().live_bytes, u64());
    EXPECT_EQ(tracker.snapshot().allocations, tracker.snapshot().deallocations);
}
TEST(FftMemory, SharedPlanIndependentWorkspaces) {
    const auto plan = FftPlan<f64>::make(usize(33)).unwrap();
    auto       run  = [&plan]() {
        auto                    scratch = Workspace<f64>::make(plan.workspace_len()).unwrap();
        array<Complex<f64>, 33> values {};
        for (int i = 0; i < 20; ++i) {
            for (auto& v : values) v = {};
            values[usize()].re = f64(1);
            if (plan.forward_in_place(values.as_mut_slice(), scratch).is_err()) return false;
            for (auto v : values)
                if ((v.re - f64(1)).abs() > f64(1e-12) || v.im.abs() > f64(1e-12)) return false;
        }
        return true;
    };
    auto first  = rstd::thread::spawn(run).unwrap();
    auto second = rstd::thread::spawn(run).unwrap();
    EXPECT_TRUE(rstd::move(first).join().unwrap());
    EXPECT_TRUE(rstd::move(second).join().unwrap());
}

template<typename T>
void capacity_limits() {
    FailState state { 0, 1 };
    using Plan = FftPlan<T, FailingAllocator>;
    auto limit = usize::MAX.to_primitive() / (16 * sizeof(T) + 2 * sizeof(usize));
    EXPECT_EQ(Plan::make_in(usize(limit + 1), FailingAllocator { &state }).unwrap_err(),
              Error::CapacityOverflow);
    // N fits the conservative bound, but its Bluestein M does not.
    EXPECT_EQ(Plan::make_in(usize(limit), FailingAllocator { &state }).unwrap_err(),
              Error::CapacityOverflow);
    EXPECT_EQ(state.calls, 0U);
    size_t power = 1;
    while (power <= limit / 2) power *= 2;
    EXPECT_EQ(Plan::make_in(usize(power), FailingAllocator { &state }).unwrap_err(),
              Error::AllocationFailed);
    EXPECT_EQ(state.calls, 1U);
}
TEST(FftMemory, CapacityChecksPrecedeAllocator) {
    capacity_limits<f32>();
    capacity_limits<f64>();
}

template<typename T>
void unaligned_objects() {
    struct alignas(16) Storage {
        T                     padding;
        array<Complex<T>, 17> values {};
    } input {}, output {};
    EXPECT_NE(reinterpret_cast<size_t>(input.values.data()) % 16, size_t(0));
    EXPECT_NE(reinterpret_cast<size_t>(output.values.data()) % 16, size_t(0));
    auto plan      = FftPlan<T>::make(usize(17), Normalization::Backward, Backend::Simd).unwrap();
    auto workspace = Workspace<T>::make(plan.workspace_len()).unwrap();
    input.values[usize()].re = T(1);
    plan.forward(input.values.as_slice(), output.values.as_mut_slice(), workspace).unwrap();
    for (auto v : output.values) {
        EXPECT_LE((v.re - T(1)).abs(), T(sizeof(T) == 4 ? 1e-5 : 1e-12));
        EXPECT_LE(v.im.abs(), T(sizeof(T) == 4 ? 1e-5 : 1e-12));
    }
}
TEST(FftMemory, ElementAlignedWithoutSimdAlignment) {
    unaligned_objects<f32>();
    unaligned_objects<f64>();
}
