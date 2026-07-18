#!/usr/bin/env python3
"""Submit a Vertex AI Supervised Fine-Tuning job for gemini-2.5-flash using
the JSONL datasets uploaded to GCS. Prints the tuning job resource name and
the tuned model's endpoint once available -- the job itself runs on Vertex,
so this script (and your laptop) can exit once it's submitted.

Usage:
    .venv/bin/python scripts/run_sft_job.py
"""
import time

import vertexai
from vertexai.tuning import sft

PROJECT = "ijcai-501207"
LOCATION = "us-central1"
TRAIN_URI = "gs://ijcai-501207-sft/car-bench/train.jsonl"
VAL_URI = "gs://ijcai-501207-sft/car-bench/val.jsonl"
BASE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME = "car-bench-agent-sft-v1"


def main():
    vertexai.init(project=PROJECT, location=LOCATION)

    sft_tuning_job = sft.train(
        source_model=BASE_MODEL,
        train_dataset=TRAIN_URI,
        validation_dataset=VAL_URI,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
        epochs=3,
    )

    print(f"Submitted tuning job: {sft_tuning_job.resource_name}")
    print("Polling until complete (safe to Ctrl-C and check back later; the job keeps running on Vertex)...")

    while not sft_tuning_job.has_ended:
        time.sleep(60)
        sft_tuning_job.refresh()
        print(f"  state={sft_tuning_job.state}")

    print("Tuning finished.")
    print(f"Tuned model endpoint: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Tuned model name:     {sft_tuning_job.tuned_model_name}")


if __name__ == "__main__":
    main()
