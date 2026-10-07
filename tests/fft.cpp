#include <rstd/test/gtest.hpp>
import liffto;
import rstd;
using namespace rstd::prelude;
using namespace liffto;
using rstd::size_t;
template<typename T>
using Vector = ::alloc::vec::Vec<T>;
static_assert(__LDBL_MANT_DIG__ > 53);
constexpr long double tau = 6.283185307179586476925286766559005768L;
struct Reference {
    long double re {}, im {};
};
template<typename T>
auto filled(size_t n) -> Vector<T> {
    auto v = Vector<T>::with_capacity(usize(n));
    for (size_t i = 0; i < n; ++i) v.push(T {});
    return v;
}
long double scale(size_t n, Normalization mode, bool inverse) {
    if (mode == Normalization::Ortho) return 1 / __builtin_sqrtl(n);
    if ((mode == Normalization::Forward && ! inverse) ||
        (mode == Normalization::Backward && inverse))
        return 1.L / n;
    return 1;
}
template<typename T>
auto dft(rstd::slice<Complex<T>> input, bool inverse, long double factor) {
    auto n      = input.len().to_primitive();
    auto result = filled<Reference>(n);
    for (size_t k = 0; k < n; ++k) {
        Reference sum;
        for (size_t j = 0; j < n; ++j) {
            auto angle = (inverse ? tau : -tau) * static_cast<long double>(j) * k / n;
            auto c = __builtin_cosl(angle), s = __builtin_sinl(angle);
            auto v = input[usize(j)];
            sum.re += v.re.to_primitive() * c - v.im.to_primitive() * s;
            sum.im += v.re.to_primitive() * s + v.im.to_primitive() * c;
        }
        result[usize(k)] = { sum.re * factor, sum.im * factor };
    }
    return result;
}
struct Accuracy {
    long double maximum {}, relative {};
};
template<typename T>
void verify(rstd::slice<Complex<T>> actual,
            rstd::slice<Reference>  expected,
            rstd::slice<Complex<T>> input,
            size_t                  m,
            long double             factor,
            Accuracy&               accuracy) {
    long double norm = 0, err = 0, denom = 0, maximum = 0;
    for (auto v : input)
        norm +=
            __builtin_sqrtl(v.re.to_primitive() * static_cast<long double>(v.re.to_primitive()) +
                            v.im.to_primitive() * static_cast<long double>(v.im.to_primitive()));
    for (usize i; i < actual.len(); ++i) {
        auto dr = actual[i].re.to_primitive() - expected[i].re,
             di = actual[i].im.to_primitive() - expected[i].im;
        auto e  = __builtin_sqrtl(dr * dr + di * di);
        if (e > maximum) maximum = e;
        err += dr * dr + di * di;
        denom += expected[i].re * expected[i].re + expected[i].im * expected[i].im;
    }
    size_t stages = 0;
    for (auto i = m; i > 1; i >>= 1) ++stages;
    if (stages == 0) stages = 1;
    long double epsilon = sizeof(T) == 4 ? 0x1p-23L : 0x1p-52L;
    auto tolerance = (sizeof(T) == 4 ? 32 : 64) * epsilon * stages * norm * __builtin_fabsl(factor);
    EXPECT_LE(maximum, tolerance);
    if (maximum > accuracy.maximum) accuracy.maximum = maximum;
    auto relative = denom == 0 ? 0 : __builtin_sqrtl(err / denom);
    if (relative > accuracy.relative) accuracy.relative = relative;
}
template<typename T>
void matrix() {
    Accuracy      accuracy;
    size_t        lengths[] = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65 };
    Normalization modes[]   = {
        Normalization::None, Normalization::Backward, Normalization::Forward, Normalization::Ortho
    };
    Backend backends[] = { Backend::Scalar, Backend::Simd, Backend::Auto };
    for (auto n : lengths)
        for (auto mode : modes)
            for (auto backend : backends) {
                auto plan      = FftPlan<T>::make(usize(n), mode, backend).unwrap();
                auto workspace = Workspace<T>::make(plan.workspace_len()).unwrap();
                auto input = filled<Complex<T>>(n), output = filled<Complex<T>>(n);
                for (int pattern = 0; pattern < 5; ++pattern) {
                    unsigned seed = 123;
                    for (size_t j = 0; j < n; ++j) {
                        Complex<T> v;
                        if (pattern == 1 && j == 0) v = { T(1), T(-0.5) };
                        if (pattern == 2) v = { T(0.75), T(-1.25) };
                        if (pattern == 3)
                            v = { T(__builtin_cosl(tau * 2 * j / n)),
                                  T(__builtin_sinl(tau * 2 * j / n)) };
                        if (pattern == 4) {
                            seed = 1664525U * seed + 1013904223U;
                            v.re = T(static_cast<int>(seed >> 16) - 32768) / T(32768);
                            seed = 1664525U * seed + 1013904223U;
                            v.im = T(static_cast<int>(seed >> 16) - 32768) / T(16384);
                        }
                        input[usize(j)] = v;
                    }
                    for (bool inverse : array<decltype(false), 2> { false, true }) {
                        auto expected = dft(input.as_slice(), inverse, scale(n, mode, inverse));
                        auto status   = inverse ? plan.inverse(input.as_slice(),
                                                               output.as_mut_slice().as_mut_ref(),
                                                               workspace)
                                                : plan.forward(input.as_slice(),
                                                               output.as_mut_slice().as_mut_ref(),
                                                               workspace);
                        ASSERT_TRUE(status.is_ok());
                        verify<T>(output.as_slice(),
                                  expected.as_slice(),
                                  input.as_slice(),
                                  plan.convolution_len().to_primitive(),
                                  scale(n, mode, inverse),
                                  accuracy);
                    }
                    ASSERT_TRUE(plan.forward(input.as_slice(),
                                             output.as_mut_slice().as_mut_ref(),
                                             workspace)
                                    .is_ok());
                    ASSERT_TRUE(plan.inverse_in_place(output.as_mut_slice().as_mut_ref(), workspace)
                                    .is_ok());
                    auto expected = filled<Reference>(n);
                    auto factor   = mode == Normalization::None ? n : 1;
                    for (usize i; i < input.len(); ++i)
                        expected[i] = {
                            input[i].re.to_primitive() * static_cast<long double>(factor),
                            input[i].im.to_primitive() * static_cast<long double>(factor)
                        };
                    verify<T>(output.as_slice(),
                              expected.as_slice(),
                              input.as_slice(),
                              plan.convolution_len().to_primitive(),
                              factor,
                              accuracy);
                }
            }
    rstd::io::println("accuracy f{} max_abs={} max_relative_l2={}",
                      u32(sizeof(T) * 8),
                      f64(accuracy.maximum),
                      f64(accuracy.relative));
}
TEST(Fft, FloatMatrix) {
    matrix<f32>();
}
TEST(Fft, DoubleMatrix) {
    matrix<f64>();
}

template<typename T>
void large_cases() {
    Accuracy accuracy;
    for (size_t n : array<decltype(size_t(1000)), 4> {
             size_t(1000), size_t(1009), size_t(4096), size_t(65536) }) {
        for (auto backend :
             array<decltype(Backend::Scalar), 2> { Backend::Scalar, Backend::Simd }) {
            auto plan      = FftPlan<T>::make(usize(n), Normalization::Backward, backend).unwrap();
            auto workspace = Workspace<T>::make(plan.workspace_len()).unwrap();
            auto input = filled<Complex<T>>(n), output = filled<Complex<T>>(n);
            for (size_t i = 0; i < n; ++i)
                input[usize(i)] = { T(static_cast<int>((i * 17) % 101) - 50) / T(50),
                                    T(static_cast<int>((i * 31) % 127) - 63) / T(64) };
            ASSERT_TRUE(
                plan.forward(input.as_slice(), output.as_mut_slice().as_mut_ref(), workspace)
                    .is_ok());
            if (n <= 1009) {
                auto expected = dft(input.as_slice(), false, 1);
                verify<T>(output.as_slice(),
                          expected.as_slice(),
                          input.as_slice(),
                          plan.convolution_len().to_primitive(),
                          1,
                          accuracy);
            }
            long double energy = 0, spectrum = 0;
            for (size_t i = 0; i < n; ++i) {
                auto x = input[usize(i)], y = output[usize(i)];
                energy += static_cast<long double>(x.re.to_primitive()) * x.re.to_primitive() +
                          static_cast<long double>(x.im.to_primitive()) * x.im.to_primitive();
                spectrum += static_cast<long double>(y.re.to_primitive()) * y.re.to_primitive() +
                            static_cast<long double>(y.im.to_primitive()) * y.im.to_primitive();
            }
            EXPECT_LE(__builtin_fabsl(spectrum / n - energy) / energy,
                      sizeof(T) == 4 ? 1e-5L : 1e-12L);
            ASSERT_TRUE(
                plan.inverse_in_place(output.as_mut_slice().as_mut_ref(), workspace).is_ok());
            auto expected = filled<Reference>(n);
            for (usize i; i < input.len(); ++i)
                expected[i] = { input[i].re.to_primitive(), input[i].im.to_primitive() };
            verify<T>(output.as_slice(),
                      expected.as_slice(),
                      input.as_slice(),
                      plan.convolution_len().to_primitive(),
                      1,
                      accuracy);
        }
    }
    rstd::io::println("large f{} max_abs={} max_relative_l2={}",
                      u32(sizeof(T) * 8),
                      f64(accuracy.maximum),
                      f64(accuracy.relative));
}
TEST(Fft, LargeLengths) {
    large_cases<f32>();
    large_cases<f64>();
}

template<typename T>
void real_matrix() {
    Accuracy accuracy;
    for (size_t n : array<decltype(size_t(1)), 9> { size_t(1),
                                                    size_t(2),
                                                    size_t(3),
                                                    size_t(7),
                                                    size_t(8),
                                                    size_t(15),
                                                    size_t(16),
                                                    size_t(33),
                                                    size_t(64) })
        for (auto mode : array<decltype(Normalization::None), 4> { Normalization::None,
                                                                   Normalization::Backward,
                                                                   Normalization::Forward,
                                                                   Normalization::Ortho })
            for (auto backend : array<decltype(Backend::Scalar), 3> {
                     Backend::Scalar, Backend::Simd, Backend::Auto }) {
                auto plan    = RealFftPlan<T>::make(usize(n), mode, backend).unwrap();
                auto scratch = Workspace<T>::make(plan.workspace_len()).unwrap();
                auto input = filled<T>(n), output = filled<T>(n);
                auto half = filled<Complex<T>>(n / 2 + 1), full = filled<Complex<T>>(n);
                for (size_t j = 0; j < n; ++j) {
                    input[usize(j)]   = T(static_cast<int>((j * 31) % 17) - 8) / T(8);
                    full[usize(j)].re = input[usize(j)];
                }
                ASSERT_TRUE(
                    plan.forward(input.as_slice(), half.as_mut_slice().as_mut_ref(), scratch)
                        .is_ok());
                auto expected = dft(full.as_slice(), false, scale(n, mode, false));
                verify<T>(half.as_slice(),
                          expected.as_slice(),
                          full.as_slice(),
                          scratch.len().to_primitive() / 2,
                          scale(n, mode, false),
                          accuracy);
                EXPECT_EQ(half[usize()].im, T {});
                if (! (n & 1)) EXPECT_EQ(half[usize(n / 2)].im, T {});
                ASSERT_TRUE(
                    plan.inverse(half.as_slice(), output.as_mut_slice().as_mut_ref(), scratch)
                        .is_ok());
                for (size_t j = 0; j < n; ++j)
                    EXPECT_LE((output[usize(j)] -
                               input[usize(j)] * T(mode == Normalization::None ? n : 1))
                                  .abs(),
                              T(sizeof(T) == 4 ? 1e-4 * n : 1e-12 * n));
                // Independent Hermitian input checks C2R without depending on R2C.
                for (size_t k = 0; k <= n / 2; ++k)
                    half[usize(k)] = { T(k + 1) / T(7), T(k) / T(9) };
                half[usize()].im = T {};
                if (! (n & 1)) half[usize(n / 2)].im = T(-0.0);
                for (size_t k = 0; k < n; ++k) {
                    auto v         = half[usize(k <= n / 2 ? k : n - k)];
                    full[usize(k)] = { v.re, k <= n / 2 ? v.im : -v.im };
                }
                expected = dft(full.as_slice(), true, scale(n, mode, true));
                ASSERT_TRUE(
                    plan.inverse(half.as_slice(), output.as_mut_slice().as_mut_ref(), scratch)
                        .is_ok());
                auto complex_output = filled<Complex<T>>(n);
                for (usize i; i < output.len(); ++i) complex_output[i].re = output[i];
                verify<T>(complex_output.as_slice(),
                          expected.as_slice(),
                          full.as_slice(),
                          scratch.len().to_primitive() / 2,
                          scale(n, mode, true),
                          accuracy);
            }
}
TEST(RealFft, FloatMatrix) {
    real_matrix<f32>();
}
TEST(RealFft, DoubleMatrix) {
    real_matrix<f64>();
}

TEST(RealFft, ConjugateSymmetry) {
    for (auto n : array<size_t, 2> { size_t(33), size_t(64) }) {
        auto plan    = FftPlan<f64>::make(usize(n)).unwrap();
        auto real    = RealFftPlan<f64>::make(usize(n)).unwrap();
        auto scratch = Workspace<f64>::make(plan.workspace_len()).unwrap();
        auto input = filled<Complex<f64>>(n), output = filled<Complex<f64>>(n);
        auto scalar = filled<f64>(n);
        auto half   = filled<Complex<f64>>(n / 2 + 1);
        for (size_t i = 0; i < n; ++i) {
            scalar[usize(i)]   = f64(i % 7) / f64(7);
            input[usize(i)].re = scalar[usize(i)];
        }
        plan.forward(input.as_slice(), output.as_mut_slice().as_mut_ref(), scratch).unwrap();
        real.forward(scalar.as_slice(), half.as_mut_slice().as_mut_ref(), scratch).unwrap();
        for (size_t k = 1; k < n; ++k) {
            EXPECT_LE((output[usize(k)].re - output[usize(n - k)].re).abs(), f64(1e-12));
            EXPECT_LE((output[usize(k)].im + output[usize(n - k)].im).abs(), f64(1e-12));
        }
        for (usize k; k < half.len(); ++k) {
            EXPECT_LE((output[k].re - half[k].re).abs(), f64(1e-12));
            EXPECT_LE((output[k].im - half[k].im).abs(), f64(1e-12));
        }
    }
}

TEST(Fft, InputScales) {
    Accuracy accuracy;
    auto     plan    = FftPlan<f32>::make(usize(65)).unwrap();
    auto     scratch = Workspace<f32>::make(plan.workspace_len()).unwrap();
    auto     input = filled<Complex<f32>>(65), output = filled<Complex<f32>>(65);
    for (auto amplitude : array<f32, 2> { f32(1e-18), f32(1e18) }) {
        for (size_t i = 0; i < 65; ++i)
            input[usize(i)] = { amplitude * f32(static_cast<int>(i % 11) - 5),
                                amplitude * f32(static_cast<int>(i % 7) - 3) };
        auto expected = dft(input.as_slice(), false, 1);
        plan.forward(input.as_slice(), output.as_mut_slice().as_mut_ref(), scratch).unwrap();
        verify<f32>(output.as_slice(),
                    expected.as_slice(),
                    input.as_slice(),
                    plan.convolution_len().to_primitive(),
                    1,
                    accuracy);
    }
}
