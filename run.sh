#!/bin/bash

echo "=> Compiling with extreme optimization flags..."
clang++ -O3 -pthread -march=native -mtune=native -flto -funroll-loops sieve.cpp -o sieve

if [ -f "./sieve" ]; then
    echo "=> Executing. Watch your thermals..."
    ./sieve
else
    echo "❌ Compilation failed."
fi
