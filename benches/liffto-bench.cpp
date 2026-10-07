import liffto;
import rstd;
import rstd.bench;
using namespace rstd::prelude;
using namespace rstd::literals;
using namespace liffto;
using namespace rstd::bench;
using rstd::size_t;
template<typename T>
using Vector = ::alloc::vec::Vec<T>;

enum class Operation
{
    Forward,
    Inverse,
    InPlace,
    RealForward,
    RealInverse,
    Plan,
    Workspace,
    Copy,
    Short
};
auto operation_name(Operation op) -> rstd::ref<rstd::str> {
    switch (op) {
    case Operation::Forward: return "forward"_str;
    case Operation::Inverse: return "inverse"_str;
    case Operation::InPlace: return "inplace-reset"_str;
    case Operation::RealForward: return "r2c"_str;
    case Operation::RealInverse: return "c2r"_str;
    case Operation::Plan: return "plan"_str;
    case Operation::Workspace: return "workspace"_str;
    case Operation::Copy: return "copy"_str;
    case Operation::Short: return "short32"_str;
    }
    return "invalid"_str;
}
auto backend_name(Backend b) -> rstd::ref<rstd::str> {
    return b == Backend::Scalar ? "scalar"_str : b == Backend::Simd ? "simd"_str : "auto"_str;
}
template<typename T>
auto filled(size_t n) -> Vector<T> {
    auto values = Vector<T>::with_capacity(usize(n));
    for (size_t i = 0; i < n; ++i) values.push(T {});
    return values;
}

template<typename T>
struct FftSession {
    size_t             n, slots, offset, cursor {};
    Operation          op;
    Backend            backend;
    FftPlan<T>         plan;
    RealFftPlan<T>     real;
    Workspace<T>       scratch;
    Vector<Complex<T>> input, output, half;
    Vector<T>          real_input, real_output, copy_re, copy_im;
    auto               source() const {
        return rstd::slice<Complex<T>>::from_raw_parts(
            input.as_slice().as_raw_ptr() + offset + cursor * n, usize(n));
    }
    auto destination() {
        return rstd::mut_ref<Complex<T>[]>::from_raw_parts(
            output.as_mut_slice().as_raw_ptr() + offset, usize(n));
    }
    void execute() {
        auto src = source();
        auto dst = destination();
        switch (op) {
        case Operation::Forward:
        case Operation::Short: plan.forward(src, dst, scratch).unwrap(); break;
        case Operation::Inverse: plan.inverse(src, dst, scratch).unwrap(); break;
        case Operation::InPlace:
            for (usize i; i < usize(n); ++i) dst[i] = src[i];
            plan.forward_in_place(dst, scratch).unwrap();
            break;
        case Operation::RealForward:
            real.forward(real_input.as_slice(), half.as_mut_slice().as_mut_ref(), scratch).unwrap();
            break;
        case Operation::RealInverse:
            real.inverse(half.as_slice(), real_output.as_mut_slice().as_mut_ref(), scratch)
                .unwrap();
            break;
        case Operation::Copy: {
            auto* re = copy_re.as_mut_slice().as_raw_ptr();
            auto* im = copy_im.as_mut_slice().as_raw_ptr();
            for (size_t i = 0; i < n; ++i) {
                re[i] = src[usize(i)].re;
                im[i] = src[usize(i)].im;
            }
            rstd::hint::black_box(copy_re.as_slice());
            rstd::hint::black_box(copy_im.as_slice());
            for (size_t i = 0; i < n; ++i) dst[usize(i)] = { re[i], im[i] };
            break;
        }
        default: break;
        }
        rstd::hint::black_box(dst);
        rstd::hint::black_box(half.as_slice());
        rstd::hint::black_box(real_output.as_slice());
        cursor = (cursor + 1) % slots;
    }
    auto check() -> Result<empty, String> {
        execute();
        return finish();
    }
    auto finish() -> Result<empty, String> {
        if (op == Operation::Plan || op == Operation::Workspace) return Ok(empty {});
        auto tolerance = T(sizeof(T) == 4 ? 1e-4 : 1e-11);
        if (op == Operation::RealForward) {
            for (auto v : half)
                if ((v.re - T(1)).abs() > tolerance || v.im.abs() > tolerance)
                    return Err("r2c check failed"_Str);
        } else if (op == Operation::RealInverse) {
            for (size_t i = 0; i < n; ++i)
                if ((real_output[usize(i)] - T(i == 0 ? 1 : 0)).abs() > tolerance)
                    return Err("c2r check failed"_Str);
        } else {
            auto dst = destination();
            for (size_t i = 0; i < n; ++i) {
                T factor = op == Operation::Inverse ? T(1) / T(n) : T(1);
                if (op == Operation::Copy) factor = i == 0 ? T(1) : T {};
                if ((dst[usize(i)].re - factor).abs() > tolerance ||
                    (dst[usize(i)].im + factor / T(2)).abs() > tolerance)
                    return Err("complex check failed"_Str);
            }
        }
        return Ok(empty {});
    }
    auto run(Bench& bench, rstd::ref<rstd::str> name) -> Result<BenchmarkResult, BenchError> {
        if (op == Operation::Plan)
            return bench.run(name, [&] {
                return FftPlan<T>::make(usize(n), Normalization::Backward, backend).unwrap();
            });
        if (op == Operation::Workspace)
            return bench.run(name, [&] {
                return Workspace<T>::make(plan.workspace_len()).unwrap();
            });
        auto      count = op == Operation::Short ? 32 : 1;
        RunConfig config;
        config.unit  = "fft"_Str;
        config.batch = f64(count);
        return bench.run(
            name,
            [&] {
                for (int i = 0; i < count; ++i) execute();
            },
            rstd::move(config));
    }
};
template<typename T>
struct FftFactory {
    size_t    n, slots, offset;
    Operation op;
    Backend   backend;
    auto      prepare() -> Result<FftSession<T>, String> {
        auto plan    = FftPlan<T>::make(usize(n), Normalization::Backward, backend).unwrap();
        auto real    = RealFftPlan<T>::make(usize(n), Normalization::Backward, backend).unwrap();
        auto scratch = Workspace<T>::make(plan.workspace_len()).unwrap();
        FftSession<T> s { n,
                          slots,
                          offset,
                          0,
                          op,
                          backend,
                          rstd::move(plan),
                          rstd::move(real),
                          rstd::move(scratch),
                          filled<Complex<T>>(n * slots + offset),
                          filled<Complex<T>>(n + offset),
                          filled<Complex<T>>(n / 2 + 1),
                          filled<T>(n),
                          filled<T>(n),
                          filled<T>(n),
                          filled<T>(n) };
        for (size_t i = 0; i < slots; ++i) s.input[usize(offset + i * n)] = { T(1), T(-0.5) };
        s.real_input[usize()] = T(1);
        for (auto& v : s.half) v = { T(1), T {} };
        return Ok(rstd::move(s));
    }
};
struct CsvReporter {
    auto report(const RunResult& result) -> Result<empty, String> {
        if (! result.outcome.is_Passed()) {
            TextReporter text;
            return text.report(result);
        }
        const auto& measurement = result.outcome.as_Passed().measurement;
        auto        s           = measurement.summary();
        rstd::io::println("summary,{},{},{},{},{}",
                          result.descriptor.id,
                          result.repetition,
                          s.median_ns_per_unit,
                          s.minimum_ns_per_unit,
                          s.maximum_ns_per_unit);
        for (const auto& epoch : measurement.measurements())
            rstd::io::println("epoch,{},{},{},{}",
                              result.descriptor.id,
                              result.repetition,
                              epoch.iterations,
                              epoch.elapsed.as_nanos());
        return Ok(empty {});
    }
};
template<typename T>
void add(Suite&    suite,
         size_t    n,
         Operation op,
         Backend   backend,
         size_t    slots  = 1,
         size_t    offset = 0) {
    auto           name = rstd::format("f{}-{}-{}-{}-slots{}-offset{}",
                                       u32(sizeof(T) * 8),
                                       usize(n),
                                       operation_name(op),
                                       backend_name(backend),
                                       usize(slots),
                                       usize(offset));
    CaseDescriptor descriptor { .id             = rstd::move(name),
                                .suite          = "fft"_Str,
                                .revision       = "1"_Str,
                                .implementation = "liffto"_Str };
    descriptor.parameters.push(Parameter { "n"_Str, ParameterValue::Unsigned(u64(n)) });
    suite.add(rstd::move(descriptor), FftFactory<T> { n, slots, offset, op, backend }).unwrap();
}
template<typename T>
void cases(Suite& suite) {
    size_t  lengths[]  = { 1, 2, 4, 8, 16, 64, 256, 1024, 4096, 65536, 15, 1000, 1009 };
    Backend backends[] = { Backend::Scalar, Backend::Simd, Backend::Auto };
    for (auto n : lengths) {
        for (auto b : backends) {
            add<T>(suite, n, Operation::Forward, b);
            add<T>(suite, n, Operation::Inverse, b);
        }
        add<T>(suite, n, Operation::Plan, Backend::Auto);
        add<T>(suite, n, Operation::Workspace, Backend::Auto);
        add<T>(suite, n, Operation::Copy, Backend::Auto);
    }
    for (size_t n : array<size_t, 3> { size_t(15), size_t(1024), size_t(1009) })
        for (auto b : backends) {
            add<T>(suite, n, Operation::RealForward, b);
            add<T>(suite, n, Operation::RealInverse, b);
        }
    add<T>(suite, 64, Operation::Short, Backend::Auto, 32, 1);
    add<T>(suite, 1024, Operation::InPlace, Backend::Auto, 1, 1);
    add<T>(suite, 65536, Operation::Forward, Backend::Auto, 64, 1);
}
int main(int argc, const char* const* argv) {
    rstd::env::args_init(argc, argv);
    RunnerOptions options;
    options.sampling.epochs            = usize(9);
    options.repeats                    = u64(3);
    options.sampling.warmup_iterations = u64(3);
    auto args                          = rstd::env::args();
    args.next();
    while (auto arg = args.next()) {
        if (*arg == "--quick"_str) {
            options.quick   = true;
            options.repeats = u64(1);
        } else if (*arg == "--id"_str) {
            auto value = args.next();
            if (value.is_none()) return 2;
            options.id = Some(rstd::move(*value));
        } else {
            rstd::io::eprintln("usage: liffto-bench [--quick] [--id exact-case-id]");
            return 2;
        }
    }
    Suite suite;
    cases<f32>(suite);
    cases<f64>(suite);
    CsvReporter reporter;
    auto        result = Runner(rstd::move(options)).run(suite, reporter);
    if (result.is_err()) {
        rstd::io::eprintln("{}", result.unwrap_err());
        return 1;
    }
    return result->passed() ? 0 : 1;
}
