#!/bin/bash

# The ?v=$RANDOM acts as a cache-buster so GitHub always serves the newest commit
CPP_URL="https://raw.githubusercontent.com/sireenyadav/sieve-benchmark/main/sieve.cpp?v=$RANDOM"

echo "=> Fetching fresh C++ payload..."
curl -sL "$CPP_URL" -o sieve.cpp

# Failsafe check
if grep -q "404: Not Found" sieve.cpp; then
    echo "❌ ERROR: GitHub returned a 404. Ensure repo is Public and branch is 'main'."
    exit 1
fi

echo "=> Installing Clang compiler..."
pkg update -y > /dev/null 2>&1
pkg install clang -y > /dev/null 2>&1

echo "=> Compiling with extreme optimization flags..."
clang++ -O3 -pthread -march=native -mtune=native -flto -funroll-loops sieve.cpp -o sieve

if [ -f "./sieve" ]; then
    echo "=> Executing. Watch your thermals..."
    ./sieve
else
    echo "❌ Compilation failed."
fi
