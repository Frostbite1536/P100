# Ticket p10-p2p-test-race: fix a race in tests/test_p2p.cpp setup (pageable cudaMemcpy then non-blocking stream)

Your current directory is a git worktree of the P100 repo (branch `p10-p2p-test-race`, based on `main`). Work only here.

## Rules
- Start every shell command with `export PATH=/home/p51/Github/P100/.venv/bin:$PATH;`. Never install anything. Build into `build/`.
- Do NOT commit or run git commands that write. The manager commits.
- Do not change what any test checks, its sizes, seeds or expectations. This ticket only fixes setup ordering.
- No network. Stay inside "Files you may touch".

## Background (manager-verified)
`p2p_buffer_reuse` fails intermittently (manager: 53/400 runs under GPU load, ~4% idle). Cause, confirmed with a standalone repro (12/300 bad → 0/300 after a device sync): the tests fill `r.src` with a **synchronous `cudaMemcpy` from pageable host memory** (`std::vector`), then read it from `r.stream`, which is created with `cudaStreamNonBlocking`. CUDA documents that a pageable host-to-device `cudaMemcpy` may return once the data is in the staging buffer, before the DMA to device memory completes, and a non-blocking stream does not synchronise with the legacy default stream. So the copies under test can read stale source data. The P2P library itself is not at fault.

## Design (manager decided)
After each pageable host-to-device `cudaMemcpy` that fills a buffer later read on a non-blocking stream (`tests/test_p2p.cpp` lines ~115, ~157, ~185), add `P100_CUDA_CHECK(cudaDeviceSynchronize());` (for the peer test, synchronise the device that owns the buffer; keep the existing device-selection calls valid) with a one-line comment explaining why (pageable cudaMemcpy may return before the DMA completes; the test streams are non-blocking). Check every other `cudaMemcpy`/`cudaMemset` (non-async) in the file for the same pattern and fix those too; list each one you changed or deliberately left.

## Files you may touch
- `tests/test_p2p.cpp`. Nothing else.

## Must pass before you finish
- `cmake -S . -B build -G Ninja && cmake --build build && ctest --test-dir build -R p2p --output-on-failure` (your sandbox may have no GPU: then the p2p tests report "not run"/skip or fail at device discovery — say which)

## Done means
Final report: the diff verbatim; each command with its result line. The manager runs the 400× loop under load.
