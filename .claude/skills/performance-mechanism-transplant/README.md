# performance-mechanism-transplant

## What this skill is for

Use this skill when a kernel, backend, or upstream PR is fast elsewhere and you
want to determine **which performance mechanisms should be transplanted into
the current production path**, rather than porting the whole backend.

The core rule is **MECHANISM TRANSPLANT, NOT BACKEND TRANSPLANT**. Read
`SKILL.md` for the method; this README is only the install / reuse guide.

## Fixed GitHub source

```text
Repository:  https://github.com/liumengjiesss/vllm-plugin-das
Branch:      docs/performance-mechanism-transplant
Skill path:  .claude/skills/performance-mechanism-transplant/
```

The repository path is the canonical source of truth. A workspace symlink (see
Option D) is a convenience and must point back to this folder.

---

## Option A — clone the branch directly

Use this on a new machine or in a new workspace that does not already have the
repository:

```bash
git clone \
  --branch docs/performance-mechanism-transplant \
  --single-branch \
  https://github.com/liumengjiesss/vllm-plugin-das.git
```

The skill is then at:

```text
vllm-plugin-das/.claude/skills/performance-mechanism-transplant/
```

If the workspace should discover it automatically, place or symlink this folder
under the workspace's `.claude/skills/`.

---

## Option B — fetch into an existing clone

Use this when the repository is already cloned. The fork remote may be named
`fork`, `origin`, or something else depending on the clone; use the actual
remote name in the commands below.

```bash
cd /path/to/vllm-plugin-das
git fetch fork docs/performance-mechanism-transplant
git log fork/docs/performance-mechanism-transplant -3 --oneline
```

Only check out the branch if you explicitly want to switch the worktree:

```bash
git checkout docs/performance-mechanism-transplant
```

Do **not** check out or reset a dirty worktree. Check `git status --short`
first. If the workspace has unrelated changes, read the files from the
remote-tracking ref instead:

```bash
git show fork/docs/performance-mechanism-transplant:.claude/skills/performance-mechanism-transplant/SKILL.md
```

---

## Option C — fetch only the skill commit

Use this when another repository should not switch branches. First identify the
latest commit that touched the skill:

```bash
git fetch fork docs/performance-mechanism-transplant
git log fork/docs/performance-mechanism-transplant -- \
  .claude/skills/performance-mechanism-transplant
```

Inspect it:

```bash
git show <commit> --stat
git show <commit> -- .claude/skills/performance-mechanism-transplant
```

If the target repository should receive the documentation commit, cherry-pick
it only after checking for unrelated changes and conflicts:

```bash
git status --short
git cherry-pick <commit>
```

The commit is documentation-only. It does not change product source, kernels,
benchmarks, or `site-packages`. If the target repository does not want the
branch history, copy the files from `git show` instead of cherry-picking.

---

## Option D — workspace symlink

If the workspace should auto-discover the skill, create a symlink under the
workspace's `.claude/skills/`. On this host the existing workspace is:

```text
/public/home/liumengjie/vllm028-work/
  vllm-plugin-das/
  .claude/skills/
```

Create or refresh the link **without overwriting an existing skill**:

```bash
SRC=/public/home/liumengjie/vllm028-work/vllm-plugin-das/.claude/skills/performance-mechanism-transplant
DEST=/public/home/liumengjie/vllm028-work/.claude/skills/performance-mechanism-transplant

if [ -e "$DEST" ] || [ -L "$DEST" ]; then
  echo "destination already exists; not overwriting: $DEST"
else
  ln -s "$SRC" "$DEST"
fi
```

Do **not** use `ln -sf`. It can silently replace a different skill or a real
directory. If the destination exists, inspect it first:

```bash
ls -la "$DEST"
readlink -f "$DEST"
```

Remove or move an obsolete link only after that inspection.

---

## Pair with senior-performance-optimization

The two skills are complementary:

```text
performance-mechanism-transplant
  = WHAT mechanism to transplant / WHAT to optimize
  = is the mechanism already present? is the backend worth porting at all?

senior-performance-optimization
  = HOW to measure / validate / account
  = baseline, correctness, gap accounting, measurement reliability, E2E
```

Recommended reading order:

1. `senior-performance-optimization/SKILL.md` — measurement and acceptance
   discipline.
2. `performance-mechanism-transplant/SKILL.md` — mechanism inventory and
   transplant decision.
3. Task-specific evidence, profiles, and source.

When the task is specifically:

> "An upstream backend/kernel/PR is fast — can we borrow it?"

read `performance-mechanism-transplant/SKILL.md` **first**, then use
`senior-performance-optimization` to run the experiments and the acceptance
gates.

Do not answer a mechanism-transplant question with a backend port, a broad
config sweep, or a single microbenchmark.

---

## Quick prompt for future use

Copy this block into a new Claude session:

```text
Read:

.claude/skills/senior-performance-optimization/SKILL.md
.claude/skills/performance-mechanism-transplant/SKILL.md

Task:
Analyze <SOURCE_KERNEL_OR_PR> for <CURRENT_PRODUCTION_PATH>.

Do not port the backend first.

Produce:

SOURCE_IMPLEMENTATION=
WHY_FAST=
REAL_OPERATING_POINT=
CURRENT_INCUMBENT_PATH=
MECHANISM_INVENTORY=
ALREADY_ABSORBED=
REMAINING_GAPS=
TOP_CANDIDATE=
EXPECTED_MS_STEP=
MINIMAL_POC=
CORRECTNESS_GATE=
AGGREGATE_GATE=
E2E_GATE=
FAILURE_CLASS=
NEXT_STEP=

Use:
MECHANISM TRANSPLANT,
not BACKEND TRANSPLANT.
```

For a fixed-shape specialization task, add:

```text
Apply references/fixed-shape-specialization.md.
Prove the real shape and selector key before proposing any config change.
```

For a rewrite or kernel-copy experiment, add:

```text
Apply references/validation-ladder.md L0-L7.
Do not read timing until L2 correctness passes against the installed kernel.
```

---

## Provenance

```text
Flagship source case:  AIOPT-1269 Step5 C1
Method:                MECHANISM_TRANSPLANT
Confirmed positive:    Marlin-style fixed-shape specialization
                       -> MTP draft BF16 MoE C1
```

Historical result:

```text
BW1100 median E2E decode   +1.11 %
BW1101 median E2E decode   +1.39 %
Paired wins                3/3
ITL                        improved (-1.98 % on BW1101)
Prefill / MTP acceptance   no regression / unchanged
Correctness                bit-exact (max_abs 0.0)
```

Important correction:

```text
This was NOT:
  "Marlin kernel +1.39 %"

It was:
  Marlin-inspired fixed-shape specialization
  implemented on the existing HCU path.
```

Also preserved in the case study:

```text
Old D0 "B reuse" positive = REFUTED
Reason: address-rewrite correctness defect
        (missing k * BLOCK_SIZE_K * stride_bk; re-read the k=0 tile).
```

The historical +1.39 % is a `CONFIRMED_POSITIVE` result, not a
`STRICT_NEW_BEST`: the Step5 project used a conservative 1.99 % bar for the
latter. That bar is a project-level label, not a universal performance
threshold.

---

## Files

```text
SKILL.md
  Core method, 10-step algorithm, decision template, stop rules.

references/mechanism-inventory.md
  Mechanism table, detection recipes, status vocabulary, worksheet.

references/fixed-shape-specialization.md
  Target definition, knob-by-knob guidance, LDS/capacity arithmetic.

references/validation-ladder.md
  L0-L7 gates, claim vocabulary, evidence package.

references/anti-patterns.md
  Recurring failure modes and the correct move for each.

references/failure-classification.md
  Eight-class taxonomy and primary/secondary assignment.

references/case-studies/step5-marlin-style-c1.md
  Flagship case: fallback root cause, rejected backend transplant, C1,
  confirmed-positive evidence, and the corrected negative lessons.
```

## Maintenance

- Edit the canonical files in this repository folder.
- If the workspace symlink exists, it follows the repository folder
  automatically; no copy is needed.
- Do not add benchmark artifacts, host-specific paths, or chat transcripts to
  this skill. The case study already records the durable evidence and the
  project paths.
- Keep `README.md` an install/reuse guide. Put method changes in `SKILL.md` or
  the relevant reference.
