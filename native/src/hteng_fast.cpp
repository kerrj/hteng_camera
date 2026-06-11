// hteng_fast — native acceleration for the hteng_camera fast path.
//
// First kernel: display-gamma encode (linear uint16 -> uint8 via a 65536-entry
// LUT), multithreaded with std::thread. It is intentionally a plain LUT gather
// compiled at -O3: the operation is memory-bandwidth bound, so portable flags
// match -march=native and there is no SIMD/SIGILL concern across CPUs.
//
// The Python side (convert.tonemap_linear) builds the LUT (it owns the tone-curve
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

// Shared threading policy: split [0, n) across T threads, single-threaded when
// the per-thread share is below the spawn-cost threshold (~256K samples).
template <typename Worker>
static void run_chunked(size_t n, int n_threads, Worker worker) {
    int T = n_threads;
    if (T <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        T = hc ? (int)hc : 4;
    }
    const size_t MIN_PER_THREAD = 1u << 18;
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

extern "C" {

// Return a small version int so Python can sanity-check the loaded binary.
int hteng_fast_version() { return 4; }

// Unpack a 12-bit *packed* Bayer buffer (hi-byte-first, 2 px per 3 bytes) into a
// uint16 plane (values 0..4095). Single-threaded on purpose: at ~0.7 ms for 5 MP
// it is 9x faster than the numpy equivalent without spawning a single thread, so
// it adds no CPU-scheduling load. Demosaic is left to cv2 (already SIMD-tuned).
//
//   p0 = (b0 << 4) | (b1 & 0x0F)
//   p1 = (b2 << 4) | (b1 >> 4)
//
// `n_pixels` must be even (2 px share each 3-byte group); a trailing odd pixel,
// if any, is left untouched.
void hteng_unpack_bayer12(const uint8_t* in, uint16_t* out, size_t n_pixels) {
    size_t pairs = n_pixels / 2;
    for (size_t i = 0; i < pairs; ++i) {
        uint8_t b0 = in[0], b1 = in[1], b2 = in[2];
        in += 3;
        out[0] = (uint16_t(b0) << 4) | (b1 & 0x0F);
        out[1] = (uint16_t(b2) << 4) | (b1 >> 4);
        out += 2;
    }
}

// Apply a 256-or-65536-entry uint8 LUT over `n` uint16 samples: out[i]=lut[in[i]].
// `lut_size` guards against out-of-range indices (clamped to lut_size-1) so a
// 12-bit LUT can be used on 16-bit data safely; pass 65536 for the no-clamp path.
// `n_threads<=0` auto-selects hardware_concurrency(). Single-threaded for small n.
void hteng_apply_lut_u16(const uint16_t* in, uint8_t* out, size_t n,
                         const uint8_t* lut, int lut_size, int n_threads) {
    const uint32_t maxidx = (lut_size > 0) ? (uint32_t)(lut_size - 1) : 65535u;
    const bool clamp = (lut_size > 0 && lut_size < 65536);

    run_chunked(n, n_threads, [=](size_t lo, size_t hi) {
        if (clamp) {
            for (size_t i = lo; i < hi; ++i) {
                uint32_t v = in[i];
                out[i] = lut[v > maxidx ? maxidx : v];
            }
        } else {
            for (size_t i = lo; i < hi; ++i)
                out[i] = lut[in[i]];
        }
    });
}

// uint16->uint16 variant of the above: out[i]=lut[in[i]], LUT has 65536 uint16
// entries (one per possible input), so there is no index clamping. This is the
// transfer-function encode path (linear uint16 -> gamma/log uint16) used before
// 10-bit HEVC encoding, where collapsing to uint8 like hteng_apply_lut_u16 would
// throw away the precision the 10-bit encoder is there to keep.
void hteng_apply_lut_u16_u16(const uint16_t* in, uint16_t* out, size_t n,
                             const uint16_t* lut, int n_threads) {
    run_chunked(n, n_threads, [=](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i)
            out[i] = lut[in[i]];
    });
}

// Per-channel variants for interleaved RGB: pixel i reads channel c through
// lut[c] (three full 65536-entry tables, concatenated R|G|B). This is how
// white balance rides the tone-curve encode for free — the per-channel gain
// is folded into each channel's table at build time (Python side), so the
// apply stays a single gather per sample, same memory traffic as the
// single-LUT kernels. `n_px` is the pixel count (samples = 3 * n_px).
void hteng_apply_lut3_u16(const uint16_t* in, uint8_t* out, size_t n_px,
                          const uint8_t* lut, int n_threads) {
    run_chunked(n_px, n_threads, [=](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            const uint16_t* p = in + 3 * i;
            uint8_t* q = out + 3 * i;
            q[0] = lut[p[0]];
            q[1] = lut[65536 + p[1]];
            q[2] = lut[2 * 65536 + p[2]];
        }
    });
}

void hteng_apply_lut3_u16_u16(const uint16_t* in, uint16_t* out, size_t n_px,
                              const uint16_t* lut, int n_threads) {
    run_chunked(n_px, n_threads, [=](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            const uint16_t* p = in + 3 * i;
            uint16_t* q = out + 3 * i;
            q[0] = lut[p[0]];
            q[1] = lut[65536 + p[1]];
            q[2] = lut[2 * 65536 + p[2]];
        }
    });
}

}  // extern "C"
