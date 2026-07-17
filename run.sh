#!/bin/bash

# Change this URL to the RAW link of the sieve.cpp file in your GitHub repo
RAW_CPP_URL="https://raw.githubusercontent.com/sireenyadav/main/sieve.cpp"

echo "=> Fetching hyper-optimized payload..."
curl -sL "$RAW_CPP_URL" -o sieve.cpp

echo "=> Installing Clang compiler (if missing)..."
pkg update -y > /dev/null 2>&1
pkg install clang -y > /dev/null 2>&1

echo "=> Compiling with extreme optimization flags..."
# -O3: Max optimization
# -march=native -mtune=native: Exploits your specific CPU's instruction set
# -flto: Link Time Optimization (cross-file optimization)
# -funroll-loops: Removes loop overhead for the SIMD counting
clang++ -O3 -pthread -march=native -mtune=native -flto -funroll-loops sieve.cpp -o sieve

echo "=> Executing. Watch your thermals..."
./sieve
