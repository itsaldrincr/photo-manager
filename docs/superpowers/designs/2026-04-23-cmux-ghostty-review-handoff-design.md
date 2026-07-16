# cmux Ghostty Review Handoff Design

Date: 2026-04-23
Status: Approved in chat, pending written-spec review

## Goal

Preserve the user's existing `cull` commands while making the review stage work
reliably when the command is launched from `cmux`.

The review UI currently depends on Kitty graphics protocol rendering for photo
previews. That works in direct Ghostty, but not reliably inside `cmux`
surfaces. The fix is to hand off the full review stage to a real Ghostty window
while keeping the rest of the command flow unchanged.

## User-Facing Behavior

- `cull --review <source>`:
  if launched from `cmux`, review opens in Ghostty and the original `cmux`
  process waits until review exits.
- `cull --review-after <source>`:
  pipeline work runs in the original terminal first; when execution reaches the
  review stage, review opens in Ghostty, and the original process waits until
  review exits.
- Other non-review commands remain unchanged.
- Invalid flag combinations must still fail before any handoff logic runs.

## Chosen Approach

Use a dedicated Ghostty handoff only for review entrypoints when the parent
process is running inside `cmux`.

The Ghostty child runs the existing full review TUI, not a reduced preview-only
mode. All keep/reject/select actions, report writes, and file moves continue to
flow through the current review codepath.

The parent process blocks until the Ghostty child exits, then resumes and
completes normally.

## Detection Rules

- Trigger handoff only for review entrypoints:
  `--review` and the review phase of `--review-after`.
- Detect `cmux` primarily via `CMUX_WORKSPACE_ID` or `CMUX_SURFACE_ID`.
- Do not rely on `TERM` or `TERM_PROGRAM` alone.
- Prevent recursion with an internal environment flag such as
  `CULL_REVIEW_HANDOFF=ghostty`.
  If the flag is present, the process must run the existing review TUI directly
  and skip handoff.

## Launch Contract

On macOS, launch a fresh Ghostty instance with `open -n -W -a Ghostty --args`.

Properties:

- `-n` creates an isolated Ghostty instance for this review session.
- `-W` blocks the parent until that Ghostty instance exits.
- The child process receives:
  - project working directory
  - review source path
  - recursion-prevention env flag
  - any additional context needed to invoke the review TUI directly

The parent must treat Ghostty launch failure as a hard error and emit a clear
message rather than silently falling back to a broken `cmux`-local review.

## Parent / Child Execution Model

### Parent

- Validate flags as it does today.
- Run any non-review work locally.
- When review begins, evaluate handoff eligibility.
- If handoff is required:
  - launch Ghostty child
  - wait for child exit
  - return success/failure based on child outcome

### Child

- Start in the same repo working directory.
- Skip handoff because the recursion-prevention env flag is set.
- Launch the existing full review TUI.
- On save/quit:
  - execute moves
  - write report
  - exit cleanly
- On quit without save:
  - preserve current existing behavior
  - exit cleanly

## Integration Points

Primary code areas expected to change:

- `src/cull/cli.py`
  - review-stage routing
  - handoff detection
  - parent/child control flow
- `src/cull/cli_review.py`
  - direct review entry helper that can be invoked by the Ghostty child
- a new helper module for:
  - `cmux` detection
  - Ghostty launch command construction
  - env handoff markers

No change is intended to the existing review TUI decision logic beyond how it
is launched.

## Error Handling

- Ghostty missing:
  fail with a direct error explaining that review handoff requires Ghostty.
- Ghostty launch failure:
  fail with launch stderr or a concise wrapped message.
- Child exits non-zero:
  parent exits non-zero.
- Parent not in `cmux`:
  run current local review flow unchanged.

## Testing Plan

### Unit tests

- `cmux` detection:
  `CMUX_WORKSPACE_ID` / `CMUX_SURFACE_ID` present vs absent
- recursion prevention:
  handoff disabled when `CULL_REVIEW_HANDOFF=ghostty`
- review routing:
  `--review` hands off only in `cmux`
- review-after routing:
  non-review pipeline runs locally, handoff occurs only at review boundary
- Ghostty launch command construction:
  includes blocking/wait semantics and cwd/env propagation

### Behavior tests

- parent review flow waits for child launcher
- launch failures propagate as clear errors
- non-review commands do not invoke handoff

## Non-Goals

- Making Kitty graphics render natively inside `cmux`
- Splitting controls in `cmux` while previews live elsewhere
- Changing review semantics, keybinds, or file-move behavior

## Open Decisions Resolved In Chat

- Full review stage is handed off, not only photo previews.
- The original `cmux` process blocks and resumes only after Ghostty review
  exits.
- The same `cull` commands are preserved.
