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

using namespace std;
using steady_clock_t = std::chrono::steady_clock;

constexpr uint32_t SEG_ODDS  = 30030;                 
constexpr uint32_t SEG_SPAN  = SEG_ODDS * 2;          
constexpr uint32_t SEG_WORDS = (SEG_ODDS + 63) / 64;  
constexpr uint32_t SMALL_LIMIT = 1'000'000;           

constexpr uint32_t MAX_BLOCKS = 8'000'000;
constexpr uint32_t BATCH_SIZE = 64;
constexpr uint32_t BLOCK_NOT_DONE = 0xFFFFFFFF; // Sentinel value

alignas(64) uint64_t template_seg[SEG_WORDS];
vector<uint32_t> base_primes;

// Replaced done_flags with an array that stores the exact prime count per block
unique_ptr<atomic<uint32_t>[]> block_counts;
atomic<uint64_t> next_block{1};   
atomic<bool> stop_now{false};

static inline void clear_bit(uint64_t* buf, uint32_t idx) {
    buf[idx >> 6] &= ~(1ULL << (idx & 63));
}

string format_num(uint64_t n) {
    string s = to_string(n);
    for (int pos = (int)s.size() - 3; pos > 0; pos -= 3) s.insert((size_t)pos, ",");
    return s;
}

uint64_t count_small_primes(uint32_t limit) {
    if (limit < 2) return 0;
    vector<uint8_t> is_prime(limit + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    uint32_t root = (uint32_t)std::sqrt((long double)limit);
    for (uint32_t p = 2; p <= root; ++p) {
        if (is_prime[p]) {
            for (uint32_t x = p * p; x <= limit; x += p) is_prime[x] = 0;
        }
    }
    uint64_t cnt = 0;
    for (uint32_t i = 2; i <= limit; ++i) cnt += is_prime[i];
    return cnt;
}

void build_template() {
    memset(template_seg, 0xFF, sizeof(template_seg));

    for (uint32_t n = 1; n < SEG_SPAN; n += 2) {
        if (n == 1) {
            clear_bit(template_seg, 0); 
            continue;
        }
        if (n % 3 == 0 || n % 5 == 0 || n % 7 == 0 || n % 11 == 0 || n % 13 == 0) {
            clear_bit(template_seg, (n - 1) >> 1);
        }
    }

    // THE FIX: Eradicate the phantom padding bits in the final 64-bit word
    uint32_t valid_bits_in_last_word = SEG_ODDS % 64;
    if (valid_bits_in_last_word > 0) {
        template_seg[SEG_WORDS - 1] &= (1ULL << valid_bits_in_last_word) - 1;
    }
}

void build_base_primes() {
    vector<uint8_t> is_prime(SMALL_LIMIT + 1, 1);
    is_prime[0] = is_prime[1] = 0;
    uint32_t root = (uint32_t)std::sqrt((long double)SMALL_LIMIT);
    for (uint32_t p = 2; p <= root; ++p) {
        if (is_prime[p]) {
            for (uint32_t x = p * p; x <= SMALL_LIMIT; x += p) is_prime[x] = 0;
        }
    }
    base_primes.reserve(100000);
    for (uint32_t p = 17; p <= SMALL_LIMIT; ++p) {
        if (is_prime[p]) base_primes.push_back(p);
    }
}

void sieve_worker() {
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
                while (idx < SEG_ODDS) {
                    clear_bit(sieve, (uint32_t)idx);
                    idx += p;
                }
            }

            uint32_t count = 0;
            for (uint32_t w = 0; w < SEG_WORDS; ++w) {
                count += (uint32_t)__builtin_popcountll(sieve[w]);
            }
            
            // Log the exact prime count to this specific block index
            block_counts[block].store(count, memory_order_release);
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto start_time = steady_clock_t::now();
    auto deadline = start_time + std::chrono::seconds(90);

    build_template();
    build_base_primes();

    // Initialize the tracking array with sentinel values
    block_counts.reset(new atomic<uint32_t>[MAX_BLOCKS]);
    for(size_t i = 0; i < MAX_BLOCKS; i++) {
        block_counts[i].store(BLOCK_NOT_DONE, memory_order_relaxed);
    }

    uint32_t block0_primes = (uint32_t)count_small_primes(SEG_SPAN - 1);
    block_counts[0].store(block0_primes, memory_order_relaxed); 

    unsigned hc = thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    unsigned num_threads = min<unsigned>(hc, 8);

    vector<thread> workers;
    workers.reserve(num_threads);

    for (unsigned i = 0; i < num_threads; ++i) {
        workers.emplace_back(sieve_worker);
    }

    while (steady_clock_t::now() < deadline) {
        this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    stop_now.store(true, memory_order_relaxed);

    for (auto& t : workers) t.join();

    // The single source of truth for the exact contiguous search space
    uint64_t exact_primes = 0;
    uint64_t completed_blocks = 0;
    
    while (completed_blocks < MAX_BLOCKS) {
        uint32_t c = block_counts[completed_blocks].load(memory_order_acquire);
        if (c == BLOCK_NOT_DONE) break; // Hard barrier: Stop at the first incomplete block
        
        exact_primes += c;
        completed_blocks++;
    }

    uint64_t exact_search_space = completed_blocks * (uint64_t)SEG_SPAN;
    auto end_time = steady_clock_t::now();
    double actual_duration = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    cout << "\n======================================================\n";
    cout << "             SEGMENTED BIT-SIEVE REPORT              \n";
    cout << "======================================================\n";
    cout << "Execution time       : " << actual_duration << " seconds\n";
    cout << "Search space limit   : " << format_num(exact_search_space) << "\n";
    cout << "Total primes found   : " << format_num(exact_primes) << "\n";
    cout << "Base primes used     : " << format_num(base_primes.size()) << "\n";
    cout << "Cores utilized       : " << num_threads << "\n";
    cout << "Numbers processed/sec: " << format_num((uint64_t)(exact_search_space / actual_duration)) << "\n";
    cout << "======================================================\n";

    return 0;
}
