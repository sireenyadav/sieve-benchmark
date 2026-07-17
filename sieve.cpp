#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <cstring> // Required for fast memset

using namespace std;
using namespace std::chrono;

const uint64_t SEGMENT_BYTES = 32768; 
const uint64_t SEGMENT_SPAN = SEGMENT_BYTES * 2; 
const uint32_t MAX_BASE = 1000000; 

atomic<uint64_t> current_block{1}; 
atomic<bool> time_up{false};
atomic<uint64_t> total_primes{0};

vector<uint32_t> base_primes;

string format_num(uint64_t n) {
    string s = to_string(n);
    int insert_pos = s.length() - 3;
    while (insert_pos > 0) {
        s.insert(insert_pos, ",");
        insert_pos -= 3;
    }
    return s;
}

void sieve_worker() {
    // alignas(64) ensures the array aligns perfectly with modern CPU cache lines.
    // This prevents cache-line tearing and speeds up SIMD vectorization.
    alignas(64) uint8_t sieve[SEGMENT_BYTES];
    uint64_t local_primes = 0;

    while (!time_up.load(memory_order_relaxed)) {
        uint64_t block = current_block.fetch_add(1, memory_order_relaxed);
        uint64_t low = block * SEGMENT_SPAN;
        uint64_t high = low + SEGMENT_SPAN;

        // Raw C memset is hardware-optimized and faster than std::fill
        memset(sieve, 1, SEGMENT_BYTES);

        for (uint32_t p : base_primes) {
            if (p == 2) continue;
            uint64_t p_sq = (uint64_t)p * p;
            if (p_sq >= high) break;

            uint64_t start = (low >= p_sq) ? low + (p - low % p) % p : p_sq;
            if (start % 2 == 0) start += p;

            // Hot loop - no branch predictions needed here
            for (uint64_t j = (start - low) / 2; j < SEGMENT_BYTES; j += p) {
                sieve[j] = 0;
            }
        }

        uint64_t count = 0;
        // The -funroll-loops flag will auto-vectorize this loop using SIMD instructions
        for (uint32_t i = 0; i < SEGMENT_BYTES; ++i) {
            count += sieve[i];
        }
        local_primes += count;
    }
    total_primes.fetch_add(local_primes, memory_order_relaxed);
}

int main() {
    cout << "Initializing base primes up to " << format_num(MAX_BASE) << "...\n";
    vector<bool> is_prime(MAX_BASE + 1, true);
    is_prime[0] = is_prime[1] = false;
    for (uint32_t p = 2; p * p <= MAX_BASE; p++) {
        if (is_prime[p]) {
            for (uint32_t i = p * p; i <= MAX_BASE; i += p) is_prime[i] = false;
        }
    }
    
    uint64_t block_0_primes = 0;
    for (uint32_t p = 2; p <= MAX_BASE; p++) {
        if (is_prime[p]) base_primes.push_back(p);
    }
    
    for(uint32_t i = 0; i < SEGMENT_SPAN; i++) {
        if(i <= MAX_BASE && is_prime[i]) block_0_primes++;
    }
    total_primes = block_0_primes;

    uint32_t num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; 
    
    cout << "Detected " << num_threads << " hardware threads.\n";
    cout << "Sieving at absolute maximum capacity for 90 seconds...\n";

    auto start_time = steady_clock::now();
    vector<thread> workers;
    for (uint32_t i = 0; i < num_threads; i++) {
        workers.emplace_back(sieve_worker);
    }

    // New progress bar supervisor loop
    int total_duration = 90;
    int last_printed_sec = -1;

    while (true) {
        auto now = steady_clock::now();
        double elapsed = duration_cast<duration<double>>(now - start_time).count();

        if (elapsed >= total_duration) break;

        int current_sec = (int)elapsed;
        // Only redraw the bar once per second to prevent I/O spam
        if (current_sec != last_printed_sec) {
            int left = total_duration - current_sec;
            float progress = (float)elapsed / total_duration;
            int bar_width = 30; // Fits well on mobile screens
            int pos = bar_width * progress;

            cout << "\r[";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) cout << "=";
                else if (i == pos) cout << ">";
                else cout << " ";
            }
            cout << "] " << current_sec << "s done / " << left << "s left" << flush;
            
            last_printed_sec = current_sec;
        }
        
        // Sleep for a short burst so we don't block the 90s cutoff
        this_thread::sleep_for(milliseconds(100)); 
    }
    time_up.store(true, memory_order_relaxed);
    
    // Clear the progress bar line so the final report prints cleanly
    cout << "\r" << string(60, ' ') << "\r";

    for (auto& w : workers) {
        w.join();
    }
    
    auto end_time = steady_clock::now();
    double actual_duration = duration_cast<duration<double>>(end_time - start_time).count();

    uint64_t max_limit = (current_block.load() - 1) * SEGMENT_SPAN;

    cout << "\n======================================================\n";
    cout << "             PEAK C++ PERFORMANCE REPORT              \n";
    cout << "======================================================\n";
    cout << "Execution time       : " << actual_duration << " seconds\n";
    cout << "Search space limit   : " << format_num(max_limit) << "\n";
    cout << "Total primes found   : " << format_num(total_primes.load()) << "\n";
    cout << "Base primes used     : " << format_num(base_primes.size()) << "\n";
    cout << "Cores utilized       : " << num_threads << "\n";
    cout << "Numbers processed/sec: " << format_num((uint64_t)(max_limit / actual_duration)) << "\n";
    cout << "======================================================\n";

    return 0;
}
