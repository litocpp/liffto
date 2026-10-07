import liffto;
int main() {
    using namespace rstd::prelude;
    using namespace liffto;
    auto                   plan    = FftPlan<f64>::make(usize(7)).unwrap();
    auto                   scratch = Workspace<f64>::make(plan.workspace_len()).unwrap();
    array<Complex<f64>, 7> values {};
    values[usize()].re = f64(1);
    if (plan.forward_in_place(values.as_mut_slice(), scratch).is_err()) return 1;
    if (plan.inverse_in_place(values.as_mut_slice(), scratch).is_err()) return 2;
    return (values[usize()].re - f64(1)).abs() < f64(1e-12) ? 0 : 3;
}
