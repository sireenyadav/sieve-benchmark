The 90-Second Silicon Sieve Challenge
The Objective
Write a program that accurately counts as many prime numbers as physically possible starting from zero, within a strict 90.0-second time limit. The program must forcefully terminate when the clock runs out and report the exact, mathematically contiguous search space and the total prime count.

The Constraints
CPU Only: Must run locally on a mobile ARM CPU (Termux or native). No GPU compute (OpenCL/Vulkan/CUDA/Metal). You are strictly limited to the processor.

No Hardcoded Databases: You cannot read from a pre-computed file of primes or fetch them from a network. You must compute them in real-time. (Algorithmic templates/wheel factorization initialized during runtime are allowed).

Strict 90-Second Cutoff: The execution must stop at 90 seconds.

Zero Guesswork: The search space must be completely contiguous. No estimating, no statistical probability, and no skipped blocks.

The Current Benchmark to Beat
The current high score was achieved using a hyper-optimized, multi-threaded C++ segmented sieve. It bypasses standard CPU math loops by utilizing a 30KB algorithmic memory template (Wheel Factorization of 3, 5, 7, 11, 13) that perfectly aligns with modern L1 cache boundaries, blasting pre-sieved bytes directly into memory via memcpy.

Language: Bare-Metal C++ (Compiled via Clang with -O3 -march=native -flto -funroll-loops)

Hardware: 8-Core Mobile ARM Processor

Execution Time: 90.069 seconds

Search Space Analyzed: 104,913,468,660 (104.9 Billion numbers)

Total Primes Found: 4,311,865,018

Throughput: 1,164,811,954 numbers/sec (1.16 Billion/sec)

The Engineering Reality
Do not attempt this in Python, Node, or Java; interpreted languages and garbage collectors will tap out at a fraction of the required speed (Python maxes out around ~189 million total primes before choking on memory allocation and the GIL).

To break 1.16 Billion operations per second, writing "clean code" is not enough. You are fighting physics. You will be severely bottle-necked by:

L1 Cache Bandwidth: How fast your CPU can physically write to its own internal memory lines.

Thermal Throttling: Keeping all 8 cores pinned at 100% will cause aggressive hardware downclocking within 20 seconds.

Instruction Pipelining: If you have branching logic (if/else statements) inside your hot loop, your pipeline will stall, and you will lose.

The Challenge
Can you engineer a more efficient algorithm? Can you exploit bitwise operations, SIMD vectorization, or tighter memory alignments to squeeze out more clock cycles?

Talk is cheap. Write the code, compile it for your architecture, and run it for 90 seconds. If you can't break 1.16 Billion numbers a second, go back to the drawing board.
