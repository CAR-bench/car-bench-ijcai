#!/usr/bin/env python3
"""
Build a Gemini SFT training JSONL from existing CAR-bench validation run
output files. Passing tasks are kept as-is (they're already correct). Failing
tasks are only included if the correction is unambiguous:
  - hallucination/disambiguation tasks with r_user_end_conversation == 0.0:
    the last assistant turn is replaced with a clear, explicit acknowledgment
    of the missing capability (never a fabricated workaround).
  - tasks where the agent claimed a completed action with no matching tool
    call in that same turn: skipped (ambiguous correction, needs human review).

Usage:
    python scripts/build_sft_dataset.py output/track_1_agent_under_test/*.json \
        --out-train scratch/train.jsonl --out-val scratch/val.jsonl
"""
import argparse
import glob
import json
import random
import sys


def trajectory_to_contents(trajectory, cutoff_idx=None):
    """Convert a CAR-bench trajectory into Gemini SFT `contents` turns.

    Tool-call and tool-result turns are dropped (not represented as
    functionCall/functionResponse parts) since the SFT target here is just
    the assistant's natural-language behavior. Dropping them can otherwise
    leave two turns of the same mapped role back-to-back (e.g. a tool-calling
    assistant turn with no text, followed eventually by another user turn),
    which Gemini's tuning API rejects -- strict user/model alternation is
    required. So consecutive same-role turns are merged (text concatenated)
    rather than each appended separately.
    """
    contents = []
    turns = trajectory if cutoff_idx is None else trajectory[:cutoff_idx]
    for turn in turns:
        role = turn.get("role")
        if role == "user":
            mapped_role, text = "user", turn.get("content") or ""
        elif role == "assistant":
            text = turn.get("content") or ""
            if not text:
                continue  # pure tool-call turn, no text to represent
            mapped_role = "model"
        else:
            continue  # tool-result turns aren't represented as SFT turns

        if contents and contents[-1]["role"] == mapped_role:
            contents[-1]["parts"][0]["text"] += "\n\n" + text
        else:
            contents.append({"role": mapped_role, "parts": [{"text": text}]})
    return contents


def build_examples(path):
    with open(path) as f:
        data = json.load(f)
    splits = data["final_result"]["detailed_results_by_split"]
    examples = []
    for split_name, results in splits.items():
        for r in results:
            traj = r.get("trajectory")
            if not traj:
                continue
            trial = r.get("trial", 0)
            passed = r["reward"] == 1.0
            if passed:
                contents = trajectory_to_contents(traj)
                # Gemini SFT requires the final turn to be "model" -- a
                # trailing unanswered "user" turn (e.g. the evaluator's
                # ###STOP### end-of-conversation marker) has to be trimmed.
                while contents and contents[-1]["role"] != "model":
                    contents.pop()
                if len(contents) >= 2:
                    examples.append({
                        "contents": contents, "task_id": r["task_id"], "label": "pass", "trial": trial,
                    })
                continue

            # Failing task: only handle the clean, unambiguous case --
            # hallucination/disambiguation end-conversation failures where
            # the fix is "say you can't do it clearly", not a guessed rewrite.
            info = (r.get("reward_info") or {}).get("info") or {}
            if info.get("r_user_end_conversation") == 0.0:
                removed = r["task"].get("removed_part")
                if not removed:
                    continue
                # Build context up to (not including) the final assistant turn
                context = trajectory_to_contents(traj, cutoff_idx=len(traj) - 1)
                if len(context) < 1:
                    continue
                # Vary the refusal phrasing -- a single fixed sentence across
                # every corrected example teaches the model to parrot one
                # canned string rather than generalize "acknowledge the gap"
                # across different phrasings/tones. Chosen deterministically
                # per (task_id, trial) so reruns of this script are stable.
                templates = [
                    "I'm not able to do that right now — that specific action "
                    "isn't something I can complete with what's available to me. "
                    "Is there something else I can help with?",
                    "Sorry, I'm not able to do that with this car right now — "
                    "that feature isn't available to me at the moment.",
                    "I don't have a way to check or handle that specific part "
                    "right now, so I can't confirm it for you. Let me know if "
                    "there's something else I can try.",
                    "That's not something I'm able to complete at the moment — "
                    "I don't want to guess or make something up, so I'll leave "
                    "it there unless you'd like help with something else.",
                    "Unfortunately I can't take care of that one right now — "
                    "it's outside what I currently have access to.",
                ]
                corrected = random.Random(f"{r['task_id']}_{trial}").choice(templates)
                if context and context[-1]["role"] == "model":
                    # cutoff_idx=-1 only drops the very last raw trajectory
                    # turn, which isn't always the assistant's final reply
                    # (e.g. trailing "###STOP###" user turns) -- replace the
                    # existing trailing model turn instead of appending a
                    # second one, to keep strict user/model alternation.
                    context[-1] = {"role": "model", "parts": [{"text": corrected}]}
                else:
                    context.append({"role": "model", "parts": [{"text": corrected}]})
                examples.append({
                    "contents": context, "task_id": r["task_id"], "label": "corrected_hallucination",
                    "removed_part": removed, "trial": trial,
                })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="Validation run JSON files (glob-expanded by shell)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    all_examples = []
    seen_task_ids = set()
    for pattern in args.inputs:
        for path in glob.glob(pattern):
            for ex in build_examples(path):
                # Include trial number AND source file so multiple trials of
                # the same task (different conversations, possibly different
                # outcomes) are all kept rather than collapsed into one --
                # that diversity is the whole point of running num_trials > 1
                # and/or re-running the harness across separate invocations.
                # Without the path, two separate runs both producing
                # trial=0,1,2 for the same task_id would wrongly look like
                # duplicates and the second run's data would be dropped.
                key = (path, ex["task_id"], ex["label"], ex.get("trial", 0))
                if key in seen_task_ids:
                    continue
                seen_task_ids.add(key)
                all_examples.append(ex)

    random.Random(args.seed).shuffle(all_examples)
    n_val = int(len(all_examples) * args.val_frac)
    val, train = all_examples[:n_val], all_examples[n_val:]

    for path, subset in [(args.out_train, train), (args.out_val, val)]:
        with open(path, "w") as f:
            for ex in subset:
                f.write(json.dumps({"contents": ex["contents"]}) + "\n")

    pass_n = sum(1 for e in all_examples if e["label"] == "pass")
    corrected_n = sum(1 for e in all_examples if e["label"] == "corrected_hallucination")
    print(f"Total examples: {len(all_examples)} ({pass_n} passing, {corrected_n} corrected)")
    print(f"Train: {len(train)} -> {args.out_train}")
    print(f"Val:   {len(val)} -> {args.out_val}")
    if corrected_n == 0:
        print("WARNING: no corrected examples found -- check reward_info parsing.", file=sys.stderr)


if __name__ == "__main__":
    main()
