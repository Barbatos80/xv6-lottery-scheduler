# xv6 Lottery and Priority Scheduler Project

This repository contains three xv6 variants used to study scheduler design and behavior:

- `xv6-lottery`: xv6 with lottery scheduling (`settickets`, `getpinfo`)
- `xv6-priority`: xv6 with ratio-based priority scheduling (`setpriority`, `getpinfo`)
- `xv6-lottery-leaky-RELU`: lottery scheduler with a fixed-point MLP advisor that periodically adjusts tickets

## Repository Layout

- `xv6-lottery/`: lottery scheduler implementation and `tickettest` user program
- `xv6-priority/`: priority scheduler implementation and `prioritytest` user program
- `xv6-lottery-leaky-RELU/`: ML-assisted lottery scheduler implementation
- `README.md` (this file): project-level usage and notes

## Scheduler Variants

### 1) Lottery Scheduler (`xv6-lottery`)

Key behavior:

- Each RUNNABLE process has a positive ticket count
- Scheduler picks a random winner in `[0, total_tickets)`
- Winning process runs for that scheduling pass
- `ticks` tracks how often each process was selected

Kernel/user interfaces:

- `int settickets(int n)` where `n >= 1`
- `int getpinfo(struct pstat *ps)`
- `struct pstat` contains `inuse[]`, `tickets[]`, `pid[]`, `ticks[]`

Test program:

- `tickettest` spawns CPU-bound children with different ticket values and prints observed CPU-share deltas.

### 2) Priority Scheduler (`xv6-priority`)

Key behavior:

- Each process has a priority value (default `50`)
- Scheduler chooses RUNNABLE process with lowest `(ticks * 100) / priority`
- Selected process `ticks` is incremented on dispatch

Kernel/user interfaces:

- `int setpriority(int priority)` with accepted range `0..200` (returns old priority)
- `int getpinfo(struct pstat *ps)`
- `struct pstat` contains `inuse[]`, `priority[]`, `pid[]`, `ticks[]`

Test program:

- `prioritytest` spawns children with configured priorities and reports per-round and aggregate scheduling share.

### 3) MLP-Assisted Lottery (`xv6-lottery-leaky-RELU`)

Key behavior:

- Preserves lottery scheduling for final process selection
- Adds an advisor (`advisor_on_timer`) that runs periodically and updates tickets
- Uses fixed-point features (run/sleep/io/wait/runnable-length) and a small MLP with leaky-ReLU hidden units
- Includes a built-in self-test harness (`nn_selftest`) for qualitative sanity checks

## Build and Run

Run these commands from the variant directory you want to use (`xv6-lottery`, `xv6-priority`, or `xv6-lottery-leaky-RELU`).

```bash
# from repo root
cd xv6-lottery        # or xv6-priority / xv6-lottery-leaky-RELU

# make sure this script is executable
chmod +x kernel/sign.pl

# build
make

# run in qemu
make qemu
```

Useful run targets:

- `make qemu-nox` (serial-only / no GUI)
- `make qemu-gdb` or `make qemu-nox-gdb` (debug mode)
- `make clean`

### Toolchain Notes

The bundled xv6 README expects an x86 ELF-capable toolchain. On non-x86/non-ELF hosts, use a cross-compiler (for example with `TOOLPREFIX=i386-jos-elf-`) per classic xv6 instructions.

## Running Scheduler Tests

After booting xv6 in QEMU, use the shell:

```sh
tickettest      # in xv6-lottery or xv6-lottery-leaky-RELU
prioritytest    # in xv6-priority
```

Both tests print CSV-style interval output and final aggregate share summaries.

## Original Project Notes (Preserved)

The previous top-level README note was:

> While in either root directory `xv6-lottery` or `xv6-priority`, execute `chmod +x kernel/sign.pl` to allow execute permissions so that xv6 may be built without errors.

That guidance is still valid and generalized in the build section above.

## Upstream xv6 Attribution and Acknowledgments

This project is based on xv6, the ANSI C re-implementation of Unix v6 for x86 multiprocessors.

From the bundled xv6 READMEs:

- Inspired by John Lions's *Commentary on UNIX 6th Edition*
- Code borrows from JOS, Plan 9, FreeBSD, and NetBSD
- Contributions acknowledged from Russ Cox, Cliff Frey, Xiao Yu, Nickolai Zeldovich, Austin Clements, Greg Price, Yandong Mao, and Hitoshi Mitake
- Copyright (2006-2007): Frans Kaashoek, Robert Morris, and Russ Cox

For additional background, see the original xv6 docs referenced in each variant's `README`.
