# zkllm-entropy agent instructions

## Plan / process relationship

Project plans (scope, phase state, deferrals) live in
https://gitlab.com/jpetrie/agent-plans. The canonical process rules for
opening plan MRs live in that repo's `WORKFLOW.md`. **Read
`/tmp/agent-plans/WORKFLOW.md` before opening any plan MR.**

This file (`CLAUDE.md` in zkllm-entropy) covers the zkllm-entropy-specific
process that layers on top of `WORKFLOW.md`.

## Claims manual

The **claims manual** is the reference document arguing that this
implementation meets the project's ZK security and performance goals.
Two files:

- `docs/proof-layer-analysis.md` — narrative / teaching layer.
- `docs/CLAIMS.md` — scannable audit index, generated from the narrative.

The manual is a reference document, **not a status tracker**. Each claim
carries a `justifiedBy` tree with leaves drawn from:

- `PAPER` — cited passage in a paper (direct quote required)
- `CODE` — pointer to file:line
- `TEST` — pointer to a test
- `ASSUMPTION` — explicitly flagged assumption
- `UNJUSTIFIED` — evidence not yet in place

No fixed hierarchy across leaf types — each claim uses whichever is the
best evidence for that particular claim. Not every leaf needs a test.

## Claims-in-sync rule

When a code change alters what a claim should say, update the claims
manual in the same commit. `main` must always present code and claims in
sync. Concretely: if a commit adds a check, removes an invariant, changes
which file enforces a property, or invalidates a `justifiedBy` leaf, the
corresponding entry in `docs/proof-layer-analysis.md` (and the
regenerated `docs/CLAIMS.md`) lands in that commit.

The only persistent markers on a claim are:

- `UNJUSTIFIED` leaves in the `justifiedBy` tree — evidence not yet in
  place.
- `⚠ needs-rewrite` on the statement — the statement itself isn't yet
  satisfactory.

There is no `status: open (blocked)` field. Claims aren't tracked —
they're either complete or visibly incomplete in place.

## Readability gate for claims

High-level claims must be understandable by the maintainer. If a claim
isn't, the resolution is either a clearer statement or a citation to a
paper the maintainer should read. The maintainer flags parts that don't
land for revision in conversation or review comments — the manual itself
does **not** carry annotations about the maintainer's reading state.

## Drift reconciliation before opening an agent-plans MR

Before opening a status or feat MR in agent-plans, do a diff-walk of
commits on zkllm-entropy `main` since the last MR, reconcile the claims
manual against those commits, and flag any residual drift in the MR
description. This is the forcing function that keeps `main` + claims
honest despite MRs being infrequent.

## Code push cadence

Push ordinary code changes directly to `main`. Do not open a PR per
change. Exceptions (confirm with the maintainer first):

- Destructive or hard-to-reverse operations (force push,
  `git reset --hard`, dependency removals/downgrades, schema changes).
- Anything covered under the agent's top-level "executing actions with
  care" guidance.

For ordinary mistakes, `git revert` on `main` is the rollback path.

## Research integrity (applies to claims manual)

- **Never attribute claims to sources without a direct quote.** If you
  can't point to the exact passage, say "I believe" or "my reasoning is"
  — never "Thaler says" or "the paper argues."
- **When you hit a contradiction you can't resolve, flag it as an open
  question.** Do not invent a resolution and present it as established
  fact. Say "I don't know how X reconciles with Y" and offer to
  research it.
- **Distinguish clearly between:** (1) directly quoted / verified
  claims, (2) your own reasoning, and (3) things you're uncertain about.
  Never let category 2 or 3 masquerade as category 1.
