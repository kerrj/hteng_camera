// hteng_fast — native acceleration for the hteng_camera fast path.
//
// First kernel: display-gamma encode (linear uint16 -> uint8 via a 65536-entry
// LUT), multithreaded with std::thread. It is intentionally a plain LUT gather
// compiled at -O3: the operation is memory-bandwidth bound, so portable flags
// match -march=native and there is no SIMD/SIGILL concern across CPUs.
//
// The Python side (convert.to_display) builds the LUT (it owns the tone-curve
// math and caching) and passes it in, so the C++ stays a pure, stateless apply
// and the output is bit-identical to the numpy path.
//
// ABI: plain C exports (extern "C") so ctypes can bind them with no name
// mangling. All buffers are caller-owned; this code allocates nothing global.

#include <cstdint>
#include <cstddef>
#include <thread>
#include <vector>
#include <algorithm>

extern "C" {

// Return a small version int so Python can sanity-check the loaded binary.
int hteng_fast_version() { return 1; }

// Apply a 256-or-65536-entry uint8 LUT over `n` uint16 samples: out[i]=lut[in[i]].
// `lut_size` guards against out-of-range indices (clamped to lut_size-1) so a
// 12-bit LUT can be used on 16-bit data safely; pass 65536 for the no-clamp path.
// `n_threads<=0` auto-selects hardware_concurrency(). Single-threaded for small n.
void hteng_apply_lut_u16(const uint16_t* in, uint8_t* out, size_t n,
                         const uint8_t* lut, int lut_size, int n_threads) {
    const uint32_t maxidx = (lut_size > 0) ? (uint32_t)(lut_size - 1) : 65535u;
    const bool clamp = (lut_size > 0 && lut_size < 65536);

    auto worker = [=](size_t lo, size_t hi) {
        if (clamp) {
            for (size_t i = lo; i < hi; ++i) {
                uint32_t v = in[i];
                out[i] = lut[v > maxidx ? maxidx : v];
            }
        } else {
            for (size_t i = lo; i < hi; ++i)
                out[i] = lut[in[i]];
        }
    };

    // Threading only pays above a threshold; below it the spawn cost dominates.
    int T = n_threads;
    if (T <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        T = hc ? (int)hc : 4;
    }
    const size_t MIN_PER_THREAD = 1u << 18;  // ~256K samples
    if ((size_t)T * MIN_PER_THREAD > n || T == 1) {
        worker(0, n);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(T);
    size_t chunk = (n + T - 1) / T;
    for (int t = 0; t < T; ++t) {
        size_t lo = (size_t)t * chunk;
        if (lo >= n) break;
        size_t hi = std::min(n, lo + chunk);
        pool.emplace_back(worker, lo, hi);
    }
    for (auto& th : pool) th.join();
}

}  // extern "C"
