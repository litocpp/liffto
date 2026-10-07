import liffto;
using namespace rstd::prelude;
using namespace liffto;

extern "C" [[gnu::noinline]]
void fft_f32(const FftPlan<f32>&           plan,
             rstd::slice<Complex<f32>>     input,
             rstd::mut_ref<Complex<f32>[]> output,
             Workspace<f32>&               scratch) {
    plan.forward(input, output, scratch).unwrap();
}
extern "C" [[gnu::noinline]]
void fft_f64(const FftPlan<f64>&           plan,
             rstd::slice<Complex<f64>>     input,
             rstd::mut_ref<Complex<f64>[]> output,
             Workspace<f64>&               scratch) {
    plan.forward(input, output, scratch).unwrap();
}
int main() {
    auto p32 = FftPlan<f32>::make(usize(37), Normalization::Backward, Backend::Simd).unwrap();
    auto p64 = FftPlan<f64>::make(usize(37), Normalization::Backward, Backend::Simd).unwrap();
    auto w32 = Workspace<f32>::make(p32.workspace_len()).unwrap();
    auto w64 = Workspace<f64>::make(p64.workspace_len()).unwrap();
    array<Complex<f32>, 37> x32 {}, y32 {};
    array<Complex<f64>, 37> x64 {}, y64 {};
    x32[usize()].re = f32(1);
    x64[usize()].re = f64(1);
    fft_f32(p32, x32.as_slice(), y32.as_mut_slice(), w32);
    fft_f64(p64, x64.as_slice(), y64.as_mut_slice(), w64);
    return (y32[usize()].re - f32(1)).abs() < f32(1e-5) &&
                   (y64[usize()].re - f64(1)).abs() < f64(1e-12)
               ? 0
               : 1;
}
