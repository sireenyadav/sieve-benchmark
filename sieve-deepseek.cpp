#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>
#include <unistd.h>
#include <sched.h>

using namespace std;
using steady_clock_t = std::chrono::steady_clock;

// ---------- Configuration (pushed to L1 limit) ----------
constexpr uint32_t SEG_ODDS  = 120120;                 // 15 KB – still fits L1
constexpr uint32_t SEG_SPAN  = SEG_ODDS * 2;           // 240240
constexpr uint32_t SEG_WORDS = (SEG_ODDS + 63) / 64;   // 1878 words ≈ 15 KB
constexpr uint32_t SMALL_LIMIT = 1'000'000;

constexpr uint32_t MAX_BLOCKS = 2'000'000;             // half again
constexpr uint32_t BATCH_SIZE = 256;
constexpr uint32_t BLOCK_NOT_DONE = 0xFFFFFFFF;

alignas(64) uint64_t template_seg[SEG_WORDS];
vector<uint32_t> base_primes;

// Per‑thread accumulation array (no atomics in hot path)
unique_ptr<uint64_t[]> block_counts_raw;
atomic<bool> stop_now{false};
atomic<uint64_t> next_block{1};

// ---------- Bit operations ----------
static inline void clear_bit(uint64_t* buf, uint32_t idx) {
    buf[idx >> 6] &= ~(1ULL << (idx & 63));
}

// ---------- NEON popcount – 32‑word unrolled ----------
static uint32_t popcount_segment(const uint64_t* data) {
    uint32_t total = 0;
    const uint64_t* end = data + SEG_WORDS;
    while (data + 32 <= end) {
        uint64x2_t v0  = vld1q_u64(data);
        uint64x2_t v1  = vld1q_u64(data+2);
        uint64x2_t v2  = vld1q_u64(data+4);
        uint64x2_t v3  = vld1q_u64(data+6);
        uint64x2_t v4  = vld1q_u64(data+8);
        uint64x2_t v5  = vld1q_u64(data+10);
        uint64x2_t v6  = vld1q_u64(data+12);
        uint64x2_t v7  = vld1q_u64(data+14);
        uint64x2_t v8  = vld1q_u64(data+16);
        uint64x2_t v9  = vld1q_u64(data+18);
        uint64x2_t v10 = vld1q_u64(data+20);
        uint64x2_t v11 = vld1q_u64(data+22);
        uint64x2_t v12 = vld1q_u64(data+24);
        uint64x2_t v13 = vld1q_u64(data+26);
        uint64x2_t v14 = vld1q_u64(data+28);
        uint64x2_t v15 = vld1q_u64(data+30);

        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v0)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v1)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v2)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v3)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v4)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v5)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v6)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v7)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v8)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v9)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v10)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v11)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v12)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v13)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v14)));
        total += vaddvq_u8(vcntq_u8(vreinterpretq_u8_u64(v15)));

        data += 32;
    }
    while (data < end)
        total += __builtin_popcountll(*data++);
    return total;
}

string format_num(uint64_t n) {
    string s = to_string(n);
    for (int pos = (int)s.size() - 3; pos > 0; pos -= 3)
        s.insert((size_t)pos, ",");
    return s;
}

uint64_t count_small_primes(uint32_t limit) {
    if (limit < 2) return 0;
    vector<uint8_t> is_prime(limit+1,1);
    is_prime[0]=is_prime[1]=0;
    uint32_t root = (uint32_t)sqrt((long double)limit);
    for (uint32_t p=2; p<=root; ++p)
        if (is_prime[p])
            for (uint32_t x=p*p; x<=limit; x+=p) is_prime[x]=0;
    uint64_t cnt=0;
    for (uint32_t i=2; i<=limit; ++i) cnt+=is_prime[i];
    return cnt;
}

void build_template() {
    memset(template_seg, 0xFF, sizeof(template_seg));
    for (uint32_t n=1; n<SEG_SPAN; n+=2) {
        if (n==1) { clear_bit(template_seg,0); continue; }
        if (n%3==0||n%5==0||n%7==0||n%11==0||n%13==0)
            clear_bit(template_seg,(n-1)>>1);
    }
    uint32_t valid_bits = SEG_ODDS % 64;
    if (valid_bits>0)
        template_seg[SEG_WORDS-1] &= (1ULL<<valid_bits)-1;
}

void build_base_primes() {
    vector<uint8_t> is_prime(SMALL_LIMIT+1,1);
    is_prime[0]=is_prime[1]=0;
    uint32_t root = (uint32_t)sqrt((long double)SMALL_LIMIT);
    for (uint32_t p=2; p<=root; ++p)
        if (is_prime[p])
            for (uint32_t x=p*p; x<=SMALL_LIMIT; x+=p) is_prime[x]=0;
    base_primes.reserve(100000);
    for (uint32_t p=17; p<=SMALL_LIMIT; ++p)
        if (is_prime[p]) base_primes.push_back(p);
}

void sieve_worker(int thread_id, int num_threads) {
    if (num_threads > 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id % num_threads, &cpuset);
        sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
    }

    alignas(64) uint64_t sieve[SEG_WORDS];

    while (!stop_now.load(memory_order_relaxed)) {
        uint64_t first = next_block.fetch_add(BATCH_SIZE, memory_order_relaxed);
        uint64_t last = min<uint64_t>(first + BATCH_SIZE, MAX_BLOCKS);

        for (uint64_t block = first; block < last; ++block) {
            if (stop_now.load(memory_order_relaxed)) break;

            uint64_t low  = block * (uint64_t)SEG_SPAN;
            uint64_t high = low + SEG_SPAN;

            memcpy(sieve, template_seg, sizeof(sieve));

            for (uint32_t p : base_primes) {
                uint64_t p2 = (uint64_t)p * (uint64_t)p;
                if (p2 >= high) break;

                uint64_t start = (low + p - 1) / p * p;
                if (start < p2) start = p2;
                if ((start & 1ULL) == 0) start += p;

                uint64_t idx = (start - low - 1) >> 1;
                constexpr uint32_t UNROLL = 8;
                if (p * UNROLL < SEG_ODDS) {
                    uint64_t idx_end = SEG_ODDS - UNROLL * (uint64_t)p;
                    while (idx <= idx_end) {
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                        clear_bit(sieve, (uint32_t)idx); idx += p;
                    }
                }
                while (idx < SEG_ODDS) {
                    clear_bit(sieve, (uint32_t)idx);
                    idx += p;
                }
            }

            // Thread‑local store (no atomic overhead)
            block_counts_raw[block] = popcount_segment(sieve);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto start_time = steady_clock_t::now();
    auto deadline = start_time + chrono::seconds(90);

    build_template();
    build_base_primes();

    block_counts_raw.reset(new uint64_t[MAX_BLOCKS]);
    memset(block_counts_raw.get(), 0xFF, MAX_BLOCKS * sizeof(uint64_t)); // sentinel -1

    // Pre‑compute block 0
    block_counts_raw[0] = count_small_primes(SEG_SPAN - 1);

    unsigned hc = thread::hardware_concurrency();
    if (hc == 0) hc = 8;
    unsigned num_threads = min<unsigned>(hc, 8);

    vector<thread> workers;
    workers.reserve(num_threads);
    for (unsigned i = 0; i < num_threads; ++i)
        workers.emplace_back(sieve_worker, i, num_threads);

    while (steady_clock_t::now() < deadline)
        this_thread::sleep_for(chrono::milliseconds(5));

    stop_now.store(true, memory_order_relaxed);
    for (auto& t : workers) t.join();

    // Merge without atomics
    uint64_t exact_primes = 0;
    uint64_t completed_blocks = 0;
    while (completed_blocks < MAX_BLOCKS) {
        uint64_t c = block_counts_raw[completed_blocks];
        if (c == 0xFFFFFFFFFFFFFFFF) break;
        exact_primes += c;
        ++completed_blocks;
    }

    uint64_t exact_search_space = completed_blocks * (uint64_t)SEG_SPAN;
    auto end_time = steady_clock_t::now();
    double actual_duration = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();

    cout << "\n======================================================\n";
    cout << "           ULTRA SEGMENTED BIT-SIEVE                 \n";
    cout << "======================================================\n";
    cout << "Execution time       : " << actual_duration << " seconds\n";
    cout << "Search space limit   : " << format_num(exact_search_space) << "\n";
    cout << "Total primes found   : " << format_num(exact_primes) << "\n";
    cout << "Base primes used     : " << format_num(base_primes.size()) << "\n";
    cout << "Threads              : " << num_threads << "\n";
    cout << "Numbers processed/sec: " << format_num((uint64_t)(exact_search_space / actual_duration)) << "\n";
    cout << "======================================================\n";

    return 0;
}
