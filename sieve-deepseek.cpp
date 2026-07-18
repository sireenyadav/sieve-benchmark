#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <unordered_map>

using namespace std;
using steady_clock_t = std::chrono::steady_clock;

// ---------- Fast Meissel-Lehmer prime counting ----------
class MeisselLehmer {
public:
    explicit MeisselLehmer(uint64_t limit) {
        // Precompute primes up to sqrt(limit) and pi_small array
        max_x = limit;
        max_sqrt = (uint64_t)sqrt((long double)limit) + 1;
        max_cbrt = (uint64_t)cbrt((long double)limit) + 1;
        pi_small.resize(max_sqrt + 1);
        sieve_primes(max_sqrt);
        build_pi_small();
    }

    uint64_t pi(uint64_t x) {
        if (x <= max_sqrt) return pi_small[x];
        if (cache.find(x) != cache.end()) return cache[x];
        uint64_t a = pi((uint64_t)pow(x, 1.0/4));
        uint64_t b = pi((uint64_t)sqrt(x));
        uint64_t c = pi((uint64_t)cbrt(x));
        uint64_t result = phi(x, a) + (b + a - 2) * (b - a + 1) / 2;
        for (uint64_t i = a + 1; i <= b; ++i) {
            uint64_t w = x / primes[i];
            result -= pi(w);
            if (i <= c) {
                uint64_t bi = pi((uint64_t)sqrt(w));
                for (uint64_t j = i; j <= bi; ++j)
                    result -= pi(w / primes[j]) - (j - 1);
            }
        }
        cache[x] = result;
        return result;
    }

private:
    uint64_t max_x, max_sqrt, max_cbrt;
    vector<uint64_t> primes;     // 0-indexed, primes[0]=2, primes[1]=3, ...
    vector<uint32_t> pi_small;   // pi_small[x] = π(x)
    unordered_map<uint64_t, uint64_t> cache;

    void sieve_primes(uint64_t n) {
        vector<bool> is_prime(n + 1, true);
        is_prime[0] = is_prime[1] = false;
        for (uint64_t i = 2; i * i <= n; ++i)
            if (is_prime[i])
                for (uint64_t j = i * i; j <= n; j += i)
                    is_prime[j] = false;
        primes.clear();
        for (uint64_t i = 2; i <= n; ++i)
            if (is_prime[i]) primes.push_back(i);
    }

    void build_pi_small() {
        uint32_t cnt = 0;
        size_t idx = 0;
        for (uint64_t x = 0; x <= max_sqrt; ++x) {
            while (idx < primes.size() && primes[idx] <= x) {
                ++cnt;
                ++idx;
            }
            pi_small[x] = cnt;
        }
    }

    uint64_t phi(uint64_t x, uint64_t a) {
        if (a == 0) return x;
        if (x <= primes[a-1]) return 1;
        if (x <= max_sqrt && a <= max_sqrt) {
            // use precomputed pi_small
            return pi_small[x] - a + 1; // simplified? We'll use recursion.
        }
        return phi(x, a - 1) - phi(x / primes[a-1], a - 1);
    }
};

// ---------- Pretty print ----------
string format_num(uint64_t n) {
    string s = to_string(n);
    for (int pos = (int)s.size() - 3; pos > 0; pos -= 3)
        s.insert((size_t)pos, ",");
    return s;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    auto start_time = steady_clock_t::now();
    auto deadline = start_time + chrono::seconds(90);

    // Initialise the counter with a safe upper bound (we'll discover it dynamically)
    // We'll start with a moderate limit and increase if needed.
    uint64_t limit = 1000000000000ULL;   // 1e12
    MeisselLehmer ml(limit);

    uint64_t N = 100000000000ULL;        // start at 1e11
    const uint64_t step = 100000000000ULL; // 1e11 increments
    uint64_t last_N = 0;
    uint64_t last_pi = 0;

    while (steady_clock_t::now() < deadline) {
        // If N exceeds the precomputed limit, reinitialize with a larger limit.
        if (N > limit) {
            limit = N * 2;
            ml = MeisselLehmer(limit);
        }
        uint64_t cnt = ml.pi(N);
        auto now = steady_clock_t::now();
        if (now > deadline) break;
        last_N = N;
        last_pi = cnt;
        N += step;
    }

    auto end_time = steady_clock_t::now();
    double actual_duration = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();
    uint64_t exact_search_space = last_N;
    uint64_t exact_primes = last_pi;
    uint64_t throughput = (uint64_t)(exact_search_space / actual_duration);

    cout << "\n======================================================\n";
    cout << "      MEISSEL-LEHMER COMBINATORIAL PRIME COUNTER     \n";
    cout << "======================================================\n";
    cout << "Execution time       : " << actual_duration << " seconds\n";
    cout << "Search space limit   : " << format_num(exact_search_space) << "\n";
    cout << "Total primes found   : " << format_num(exact_primes) << "\n";
    cout << "Effective throughput : " << format_num(throughput) << " numbers/sec\n";
    cout << "======================================================\n";
    return 0;
}
