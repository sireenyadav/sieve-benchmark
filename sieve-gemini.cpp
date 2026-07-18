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

// ---------- Configuration ----------
constexpr uint32_t SEG_ODDS  = 360360;                 
constexpr uint32_t SEG_SPAN  = SEG_ODDS * 2;           
constexpr uint32_t SEG_WORDS = (SEG_ODDS + 63) / 64;   
constexpr uint32_t SMALL_LIMIT = 1'000'000;

constexpr uint32_t MAX_BLOCKS = 2'000'000;
constexpr uint32_t BATCH_SIZE = 256;
constexpr uint64_t BLOCK_NOT_DONE = 0xFFFFFFFFFFFFFFFFULL;

// Page-aligned to prevent OS TLB thrashing
alignas(4096) uint64_t template_seg[SEG_WORDS];
vector<uint32_t> base_primes;
vector<uint32_t> small_primes;          
vector<uint32_t> large_primes;          
vector<uint32_t> large_step_words;      
vector<uint32_t> large_bit_steps;       

unique_ptr<uint64_t[]> block_counts_raw;
atomic<bool> stop_now{false};
atomic<uint64_t> next_block{1};

// ---------- Inline bit clear ----------
static inline void clear_bit(uint64_t* buf, uint32_t idx) {
    buf[idx >> 6] &= ~(1ULL << (idx & 63));
}

// ---------- Assembly copy template (128 bytes/iter) ----------
static inline void copy_template(uint64_t* __restrict dest, const uint64_t* __restrict src) {
    constexpr uint32_t words = SEG_WORDS;
    uint32_t blocks = words / 16;
    uint32_t remainder = words % 16;

    if (blocks > 0) {
        asm volatile(
            "mov x3, %[cnt]\n"
            "1:\n"
            "ldp q0, q1, [%[src]], #32\n"
            "ldp q2, q3, [%[src]], #32\n"
            "ldp q4, q5, [%[src]], #32\n"
            "ldp q6, q7, [%[src]], #32\n"
            "stp q0, q1, [%[dest]], #32\n"
            "stp q2, q3, [%[dest]], #32\n"
            "stp q4, q5, [%[dest]], #32\n"
            "stp q6, q7, [%[dest]], #32\n"
            "subs x3, x3, #1\n"
            "b.ne 1b\n"
            : [src]"+r"(src), [dest]"+r"(dest)
            : [cnt]"r"((uint64_t)blocks)
            : "x3", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
        );
    }
    if (remainder) {
        for (uint32_t i = 0; i < remainder; ++i) dest[i] = src[i];
    }
}

// ---------- Assembly popcount ----------
static uint32_t popcount_asm(const uint64_t* data) {
    uint32_t total = 0;
    constexpr uint32_t words = SEG_WORDS;
    uint32_t full_blocks = words / 64;
    uint32_t remainder = words % 64;

    if (full_blocks > 0) {
        asm volatile(
            "movi v16.16b, #0\n"
            "movi v17.16b, #0\n"
            "movi v18.16b, #0\n"
            "movi v19.16b, #0\n"
            "mov x2, %[cnt]\n"
            "1:\n"
            "ldp q0, q1, [%[data], #0]\n"
            "ldp q2, q3, [%[data], #32]\n"
            "ldp q4, q5, [%[data], #64]\n"
            "ldp q6, q7, [%[data], #96]\n"
            "ldp q8, q9, [%[data], #128]\n"
            "ldp q10, q11, [%[data], #160]\n"
            "ldp q12, q13, [%[data], #192]\n"
            "ldp q14, q15, [%[data], #224]\n"
            "ldp q20, q21, [%[data], #256]\n"
            "ldp q22, q23, [%[data], #288]\n"
            "ldp q24, q25, [%[data], #320]\n"
            "ldp q26, q27, [%[data], #352]\n"
            "ldp q28, q29, [%[data], #384]\n"
            "ldp q30, q31, [%[data], #416]\n"
            "cnt v0.16b, v0.16b\n"
            "cnt v1.16b, v1.16b\n"
            "cnt v2.16b, v2.16b\n"
            "cnt v3.16b, v3.16b\n"
            "cnt v4.16b, v4.16b\n"
            "cnt v5.16b, v5.16b\n"
            "cnt v6.16b, v6.16b\n"
            "cnt v7.16b, v7.16b\n"
            "cnt v8.16b, v8.16b\n"
            "cnt v9.16b, v9.16b\n"
            "cnt v10.16b, v10.16b\n"
            "cnt v11.16b, v11.16b\n"
            "cnt v12.16b, v12.16b\n"
            "cnt v13.16b, v13.16b\n"
            "cnt v14.16b, v14.16b\n"
            "cnt v15.16b, v15.16b\n"
            "cnt v20.16b, v20.16b\n"
            "cnt v21.16b, v21.16b\n"
            "cnt v22.16b, v22.16b\n"
            "cnt v23.16b, v23.16b\n"
            "cnt v24.16b, v24.16b\n"
            "cnt v25.16b, v25.16b\n"
            "cnt v26.16b, v26.16b\n"
            "cnt v27.16b, v27.16b\n"
            "cnt v28.16b, v28.16b\n"
            "cnt v29.16b, v29.16b\n"
            "cnt v30.16b, v30.16b\n"
            "cnt v31.16b, v31.16b\n"
            "add v0.16b, v0.16b, v1.16b\n"
            "add v2.16b, v2.16b, v3.16b\n"
            "add v4.16b, v4.16b, v5.16b\n"
            "add v6.16b, v6.16b, v7.16b\n"
            "add v8.16b, v8.16b, v9.16b\n"
            "add v10.16b, v10.16b, v11.16b\n"
            "add v12.16b, v12.16b, v13.16b\n"
            "add v14.16b, v14.16b, v15.16b\n"
            "add v0.16b, v0.16b, v2.16b\n"
            "add v4.16b, v4.16b, v6.16b\n"
            "add v8.16b, v8.16b, v10.16b\n"
            "add v12.16b, v12.16b, v14.16b\n"
            "add v0.16b, v0.16b, v4.16b\n"
            "add v8.16b, v8.16b, v12.16b\n"
            "add v16.16b, v16.16b, v0.16b\n"
            "add v17.16b, v17.16b, v8.16b\n"
            "add v20.16b, v20.16b, v21.16b\n"
            "add v22.16b, v22.16b, v23.16b\n"
            "add v24.16b, v24.16b, v25.16b\n"
            "add v26.16b, v26.16b, v27.16b\n"
            "add v28.16b, v28.16b, v29.16b\n"
            "add v30.16b, v30.16b, v31.16b\n"
            "add v20.16b, v20.16b, v22.16b\n"
            "add v24.16b, v24.16b, v26.16b\n"
            "add v28.16b, v28.16b, v30.16b\n"
            "add v20.16b, v20.16b, v24.16b\n"
            "add v28.16b, v28.16b, v28.16b\n" 
            "add v18.16b, v18.16b, v20.16b\n"
            "add v19.16b, v19.16b, v28.16b\n"
            "add %[data], %[data], #512\n"
            "subs x2, x2, #1\n"
            "b.ne 1b\n"
            "uaddlv h16, v16.16b\n"
            "uaddlv h17, v17.16b\n"
            "uaddlv h18, v18.16b\n"
            "uaddlv h19, v19.16b\n"
            "add v16.4h, v16.4h, v17.4h\n"
            "add v18.4h, v18.4h, v19.4h\n"
            "add v16.4h, v16.4h, v18.4h\n"
            "smov %w[out], v16.h[0]\n"
            : [out]"=r"(total), [data]"+r"(data)
            : [cnt]"r"((uint64_t)full_blocks)
            : "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
              "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
        );
    }
    for (uint32_t i = 0; i < remainder; ++i) total += __builtin_popcountll(data[i]);
    return total;
}

// ---------- Formatting & Initialization ----------
string format_num(uint64_t n) {
    string s = to_string(n);
    for (int pos = (int)s.size() - 3; pos > 0; pos -= 3) s.insert((size_t)pos, ",");
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
        if (n%3==0||n%5==0||n%7==0||n%11==0||n%13==0) clear_bit(template_seg,(n-1)>>1);
    }
    uint32_t valid_bits = SEG_ODDS % 64;
    if (valid_bits>0) template_seg[SEG_WORDS-1] &= (1ULL<<valid_bits)-1;
}

void build_base_primes() {
    vector<uint8_t> is_prime(SMALL_LIMIT+1,1);
    is_prime[0]=is_prime[1]=0;
    uint32_t root = (uint32_t)sqrt((long double)SMALL_LIMIT);
    for (uint32_t p=2; p<=root; ++p)
        if (is_prime[p])
            for (uint32_t x=p*p; x<=SMALL_LIMIT; x+=p) is_prime[x]=0;

    for (uint32_t p=17; p<=SMALL_LIMIT; ++p) {
        if (is_prime[p]) {
            if (p < 64) small_primes.push_back(p);
            else {
                large_primes.push_back(p);
                large_step_words.push_back(p / 64);
                large_bit_steps.push_back(p % 64);
            }
        }
    }
}

// ---------- The 4-Billion Engine (Stateful Zero-Division Loop) ----------
void sieve_worker(int thread_id, int num_threads) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % num_threads, &cpuset);
    sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);

    alignas(4096) uint64_t sieve[SEG_WORDS];

    vector<uint64_t> small_idx(small_primes.size(), 0);
    vector<uint64_t> large_idx(large_primes.size(), 0);

    while (!stop_now.load(memory_order_relaxed)) {
        uint64_t first = next_block.fetch_add(BATCH_SIZE, memory_order_relaxed);
        uint64_t last = min<uint64_t>(first + BATCH_SIZE, MAX_BLOCKS);
        if (first >= MAX_BLOCKS) break;

        uint64_t batch_low = first * (uint64_t)SEG_SPAN;
        uint64_t batch_high = batch_low + (uint64_t)SEG_SPAN * (last - first);

        // Precompute divisions EXACTLY ONCE per 256 blocks
        size_t small_active = 0;
        for (size_t i = 0; i < small_primes.size(); ++i) {
            uint64_t p = small_primes[i];
            uint64_t p2 = p * p;
            if (p2 >= batch_high) break;
            small_active++;
            uint64_t start = (batch_low + p - 1) / p * p;
            if (start < p2) start = p2;
            if ((start & 1ULL) == 0) start += p;
            small_idx[i] = (start - batch_low - 1) >> 1;
        }

        size_t large_active = 0;
        for (size_t i = 0; i < large_primes.size(); ++i) {
            uint64_t p = large_primes[i];
            uint64_t p2 = p * p;
            if (p2 >= batch_high) break;
            large_active++;
            uint64_t start = (batch_low + p - 1) / p * p;
            if (start < p2) start = p2;
            if ((start & 1ULL) == 0) start += p;
            large_idx[i] = (start - batch_low - 1) >> 1;
        }

        // The Hot Loop (Absolutely ZERO divisions inside)
        for (uint64_t block = first; block < last; ++block) {
            if (stop_now.load(memory_order_relaxed)) break;

            copy_template(sieve, template_seg);

            for (size_t i = 0; i < small_active; ++i) {
                uint32_t p = small_primes[i];
                uint64_t idx = small_idx[i];
                
                if (idx < SEG_ODDS) {
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
                small_idx[i] = idx - SEG_ODDS; // Carry forward to next block cleanly
            }

            for (size_t i = 0; i < large_active; ++i) {
                uint64_t p = large_primes[i];
                uint64_t idx = large_idx[i];
                
                if (idx >= SEG_ODDS) {
                    large_idx[i] = idx - SEG_ODDS;
                    continue;
                }

                uint64_t word = idx >> 6;
                uint64_t mask = 1ULL << (idx & 63);
                uint32_t step = large_step_words[i];
                uint32_t bstep = large_bit_steps[i];
                uint64_t p8 = p << 3;

                if (bstep == 0) {
                    while (word + 8*step < SEG_WORDS) {
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        sieve[word] &= ~mask; word += step;
                        idx += p8;
                    }
                    while (word < SEG_WORDS) {
                        sieve[word] &= ~mask; word += step;
                        idx += p;
                    }
                    large_idx[i] = idx - SEG_ODDS;
                } else {
                    while (word + 8*step < SEG_WORDS) {
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        sieve[word] &= ~mask; mask = __builtin_rotateleft64(mask, bstep); word += step;
                        idx += p8;
                    }
                    while (word < SEG_WORDS) {
                        sieve[word] &= ~mask; word += step; mask = __builtin_rotateleft64(mask, bstep);
                        idx += p;
                    }
                    large_idx[i] = idx - SEG_ODDS;
                }
            }

            block_counts_raw[block] = popcount_asm(sieve);
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
    memset(block_counts_raw.get(), 0xFF, MAX_BLOCKS * sizeof(uint64_t));

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

    uint64_t exact_primes = 0;
    uint64_t completed_blocks = 0;
    while (completed_blocks < MAX_BLOCKS) {
        uint64_t c = block_counts_raw[completed_blocks];
        if (c == BLOCK_NOT_DONE) break;
        exact_primes += c;
        ++completed_blocks;
    }

    uint64_t exact_search_space = completed_blocks * (uint64_t)SEG_SPAN;
    auto end_time = steady_clock_t::now();
    double actual_duration = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();

    cout << "\n======================================================\n";
    cout << "      ZERO-DIVISION STATEFUL BATCHING (4B TARGET)   \n";
    cout << "======================================================\n";
    cout << "Execution time       : " << actual_duration << " seconds\n";
    cout << "Search space limit   : " << format_num(exact_search_space) << "\n";
    cout << "Total primes found   : " << format_num(exact_primes) << "\n";
    cout << "Base primes used     : " << format_num(small_primes.size() + large_primes.size()) << "\n";
    cout << "  (small 17..63)     : " << small_primes.size() << "\n";
    cout << "  (large >=64)       : " << large_primes.size() << "\n";
    cout << "Threads              : " << num_threads << "\n";
    cout << "Numbers processed/sec: " << format_num((uint64_t)(exact_search_space / actual_duration)) << "\n";
    cout << "======================================================\n";

    return 0;
}
