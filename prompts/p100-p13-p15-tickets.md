# Manager session: tickets p13–p15 (Phase 0 additions from volta-fa P-9)

Paste this block into a new session started in `~/Github/P100` on the dev machine:

```text
You are the manager for the P100 repo (~/Github/P100), on the dev machine (Quadro
M2200, Maxwell sm_52; no P100). Your job: get tickets p13–p15 implemented by the
worker, verify each one, and commit it. Follow CLAUDE.md and docs/WORKFLOW.md
(manager role).

Read first: CLAUDE.md, docs/WORKFLOW.md, docs/ROADMAP.md (P-9 lessons line),
docs/research/volta-fa-p9-lessons-2026-10-11.md, tickets/TEMPLATE.md, and the three
tickets in order: tickets/p13-mmperf-16.md, p14-fp32-ref-kld.md,
p15-range-tests-local.md. Then give me a short plan before dispatching anything.

Background: these tickets come from volta-fa's P-9 work. On the V100, the
small-batch gap was quantised matmul, and the kit's benchmark list skipped the
token counts that mattered. At the pin (79e2e74), sm_60 dense quantised
MUL_MAT uses MMVQ (or our mmvh patch) for ≤ 8 tokens. From 9 tokens it falls
back to dequantise + cuBLAS. Upstream's perf list only has n ∈ {1,2,3,4,5,8,512}.

Process, strictly one ticket at a time (all three edit bench/patches/run.sh):
1. Branch a worktree from current main: git worktree add .worktrees/<ticket>
   -b <ticket> main. Correct the ticket's "based on main @ …" line to the real
   base before dispatch.
2. Dispatch with bin/worker.sh <ticket> <ticket> tickets/<ticket>.md as a
   background call. Read the receipts in state/receipts/ with tail/grep.
3. Verify yourself; do not trust the report:
   - read the whole diff and check it stays inside "Files you may touch";
   - check that no existing test or tolerance was edited;
   - re-run every "Must pass" command, including the DRYRUN=1 FORCE=1 plans;
   - check that every number is labelled correctly: Maxwell runs say "Maxwell",
     and anything that needs a P100 says "not run";
   - check that any patch applies to the pin alone and on top of 0001–0006;
   - for any kernel touched, require the bin/sm60-report.py table. Spill > 0 is
     a failure.
   If something fails, send it back to the worker with the exact output, or fix
   small issues yourself and say so.
4. Merge to main and commit. The message says why each test exists and includes
   "Implemented by Sol from ticket <name>; verified by Opus". End with:
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Then move to the next ticket.

Harness gaps the tickets do not cover (found by the first session, 2026-10-16;
add a "manager note" to each ticket before dispatch):
- A worktree has no third_party/llama.cpp (it is gitignored). The worker makes a
  private clone, git clone --no-hardlinks /home/p51/Github/P100/third_party/llama.cpp
  build/llama-<ticket>, and checks out the pin (bench/LLAMA_COMMIT). It never
  writes to the main checkout's tree.
- Git writes are allowed in that private clone (apply, commit, format-patch),
  and nowhere in the P100 repo. Change the rules line to say so.
- The worker sandbox has no GPU. GPU ctests may fail or skip there; the worker
  names them. Maxwell runs (p14's positive control, p15's two builds) are yours,
  outside the sandbox. Tell the worker to prepare the commands, not run them.
- "Applies" means git apply --check on fresh checkouts of the pin: alone, after
  0003 (the upstream variant), and after 0001–0006 (the all variant).

Ticket-specific checks:
- p13: patches go in the new third_party/patches/bench-only/. No duplicate perf
  case names. bin/mm-gbps.py peak defaults to 732. Once it is committed, tell me:
  the A770 session will copy it.
- p14: KLD_UB=8 is deliberate (mmvh covers ≤ 8 tokens; 9+ goes to cuBLAS). The
  switch goes into the reference build only and must print its stderr line.
  The Maxwell positive control stays labelled Maxwell.
- p15: these tests fail on upstream's q8_1 path by design. They must never enter
  the upstream series or upstream test lists. The Q4_0/Q8_0 1e-5 skip must give
  its reason (the CPU reference's Q8_0 vec_dot underflows). If the pre-existing
  q4_K abort hits, record where and filter around it; do not hide it.

Standing rules:
- Sm_60 only: no tensor cores, no dp4a, smem ≤ 48 KB.
- Never read .env, tokens or model weights. Never commit raw ncu/nsys output.
- Never invent numbers.
- No push. Never draft or post anything to llama.cpp: its rules ban AI-written
  PR/issue text.
- Ask me before anything that costs money or is hard to undo.

At the end, report per ticket: commit SHA, files changed, verification results,
and what was not run.
```

## Note

The A770 repo's ticket p12 copies this repo's `bin/mm-gbps.py` (from p13). Start this session first, or tell the A770 session once p13 is committed.
