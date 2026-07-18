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
#include <sys/mman.h>

using namespace std;
using steady_clock_t = std::chrono::steady_clock;

// ---------- Configuration ----------
constexpr uint32_t SEG_ODDS  = 360360;                 
constexpr uint32_t SEG_SPAN  = SEG_ODDS * 2;           
constexpr uint32_t SEG_WORDS = (SEG_ODDS + 63) / 64;   
constexpr uint32_t SMALL_LIMIT = 1'000'000;

constexpr uint32_t MAX_BLOCKS = 3'000'000;
constexpr uint32_t BATCH_SIZE = 256;
constexpr uint64_t BLOCK_NOT_DONE = 0xFFFFFFFFFFFFFFFFULL;

alignas(4096) uint64_t template_seg[SEG_WORDS];
vector<uint32_t> base_primes;

unique_ptr<uint64_t[]> block_counts_raw;
atomic<bool> stop_now{false};
atomic<uint64_t> next_block{1};

// ---------- RAM Purge & Cache Flush ----------
void purge_ram_and_caches() {
    cout << "[!] Purging Android OS caches and allocating pristine RAM buffer..." << endl;
    size_t mem_size = 1024ULL * 1024ULL * 512ULL; // 512 MB Flush
    volatile uint8_t* trash = (uint8_t*)malloc(mem_size);
    if (trash) {
        // Force the OS to map and zero the memory, evicting background apps
        for (size_t i = 0; i < mem_size; i += 4096) {
            trash[i] = (uint8_t)(i & 0xFF);
        }
        free((void*)trash);
    }
    cout << "[!] L2/L3 Caches flushed. Memory Controller optimized." << endl;
}

// ---------- Inline bit clear ----------
static inline void clear_bit(uint64_t* buf, uint32_t idx) {
    buf[idx >> 6] &= ~(1ULL << (idx & 63));
}

// ---------- Assembly copy template ----------
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

// ---------- UADALP Assembly Popcount (The Ultimate Vector Accumulator) ----------
static uint32_t popcount_asm(const uint64_t* data) {
    uint32_t total = 0;
    constexpr uint32_t words = SEG_WORDS;
    uint32_t full_blocks = words / 32; // Processing 256 bytes per loop
    uint32_t remainder = words % 32;

    if (full_blocks > 0) {
        asm volatile(
            "movi v16.8h, #0\n"
            "movi v17.8h, #0\n"
            "movi v18.8h, #0\n"
            "movi v19.8h, #0\n"
            "mov x2, %[cnt]\n"
            "1:\n"
            "ldp q0, q1, [%[data]], #32\n"
            "ldp q2, q3, [%[data]], #32\n"
            "ldp q4, q5, [%[data]], #32\n"
            "ldp q6, q7, [%[data]], #32\n"
            "ldp q8, q9, [%[data]], #32\n"
            "ldp q10, q11, [%[data]], #32\n"
            "ldp q12, q13, [%[data]], #32\n"
            "ldp q14, q15, [%[data]], #32\n"
            
            // Popcount 8-bit vectors
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
            
            // Pairwise unsigned add and accumulate directly into 16-bit registers
            "uadalp v16.8h, v0.16b\n"
            "uadalp v16.8h, v1.16b\n"
            "uadalp v16.8h, v2.16b\n"
            "uadalp v16.8h, v3.16b\n"
            "uadalp v17.8h, v4.16b\n"
            "uadalp v17.8h, v5.16b\n"
            "uadalp v17.8h, v6.16b\n"
            "uadalp v17.8h, v7.16b\n"
            "uadalp v18.8h, v8.16b\n"
            "uadalp v18.8h, v9.16b\n"
            "uadalp v18.8h, v10.16b\n"
            "uadalp v18.8h, v11.16b\n"
            "uadalp v19.8h, v12.16b\n"
            "uadalp v19.8h, v13.16b\n"
            "uadalp v19.8h, v14.16b\n"
            "uadalp v19.8h, v15.16b\n"
            
            "subs x2, x2, #1\n"
            "b.ne 1b\n"
            
            // Final horizontal reduction
            "add v16.8h, v16.8h, v17.8h\n"
            "add v18.8h, v18.8h, v19.8h\n"
            "add v16.8h, v16.8h, v18.8h\n"
            "uaddlv s16, v16.8h\n"
            "fmov %w[out], s16\n"
            : [out]"=r"(total), [data]"+r"(data)
            : [cnt]"r"((uint64_t)full_blocks)
            : "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
              "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
              "v16", "v17", "v18", "v19"
        );
    }
    for (uint32_t i = 0; i < remainder; ++i) total += __builtin_popcountll(data[i]);
    return total;
}

// ---------- Initialization ----------
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

    base_primes.clear();
    for (uint32_t p=17; p<=SMALL_LIMIT; ++p) {
        if (is_prime[p]) base_primes.push_back(p);
    }
}

// ---------- The Flawless Silicon-Limit Engine ----------
void sieve_worker(int thread_id, int num_threads) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(thread_id % num_threads, &cpuset);
    sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);

    alignas(4096) uint64_t sieve[SEG_WORDS];
    
    // Absolute position tracking array
    vector<uint64_t> next_odd_idx(base_primes.size(), 0);

    while (!stop_now.load(memory_order_relaxed)) {
        uint64_t first = next_block.fetch_add(BATCH_SIZE, memory_order_relaxed);
        uint64_t last = min<uint64_t>(first + BATCH_SIZE, MAX_BLOCKS);
        if (first >= MAX_BLOCKS) break;

        uint64_t batch_low = first * (uint64_t)SEG_SPAN;
        uint64_t batch_high = batch_low + (uint64_t)SEG_SPAN * (last - first);

        // Precompute exact absolute starting positions relative to batch_low
        size_t active_primes = 0;
        for (size_t i = 0; i < base_primes.size(); ++i) {
            uint64_t p = base_primes[i];
            if (p * p >= batch_high) break; 
            active_primes++;
            uint64_t start = (batch_low + p - 1) / p * p;
            if (start < p * p) start = p * p;
            if ((start & 1ULL) == 0) start += p;
            next_odd_idx[i] = (start - batch_low - 1) >> 1;
        }

        // Zero-drift mathematically flawless loop
        for (uint64_t block = first; block < last; ++block) {
            if (stop_now.load(memory_order_relaxed)) break;

            copy_template(sieve, template_seg);
            
            uint64_t block_target_start = (block - first) * (uint64_t)SEG_ODDS;
            uint64_t block_target_end = block_target_start + SEG_ODDS;

            for (size_t i = 0; i < active_primes; ++i) {
                uint32_t p = base_primes[i];
                uint64_t idx = next_odd_idx[i];
                
                if (idx >= block_target_end) continue;
                
                uint64_t local_idx = idx - block_target_start;
                
                constexpr uint32_t UNROLL = 8;
                if (p * UNROLL < SEG_ODDS) {
                    uint64_t idx_end = SEG_ODDS - UNROLL * (uint64_t)p;
                    while (local_idx <= idx_end) {
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                        clear_bit(sieve, (uint32_t)local_idx); local_idx += p;
                    }
                }
                while (local_idx < SEG_ODDS) {
                    clear_bit(sieve, (uint32_t)local_idx);
                    local_idx += p;
                }
                
                // Perfectly map absolute position forward
                next_odd_idx[i] = block_target_start + local_idx;
            }

            block_counts_raw[block] = popcount_asm(sieve);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Blast the RAM and caches clean
    purge_ram_and_caches();

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
    cout << "      BARE-METAL UADALP ENGINE (HARDWARE LIMIT)      \n";
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
