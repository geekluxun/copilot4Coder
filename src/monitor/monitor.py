import time
from threading import Thread

import psutil
import torch
import wandb
from transformers import TrainingArguments

project_name = "CodeCopilot"

MONITOR_INTERVAL = 30


def init_monitor(training_args: TrainingArguments):
    wandb.init(
        project=project_name,
        mode="offline",
        config=training_args.to_dict(),
        allow_val_change=True,
        reinit=True
    )
    # 启动资源监控
    monitor = ResourceMonitor(interval=MONITOR_INTERVAL)
    monitor.start()


class ResourceMonitor:
    def __init__(self, interval=MONITOR_INTERVAL):
        self.interval = interval
        self.running = False

    def start(self):
        self.running = True
        self.thread = Thread(target=self._monitor, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def _monitor(self):
        while self.running:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()

            print("\n=== Resource Monitor ===")
            print(f"CPU Usage: {cpu_percent}%")
            print(f"Memory Used: {memory.percent}%")
            print(f"Available Memory: {memory.available / (1024 * 1024 * 1024):.2f} GB")

            if torch.backends.mps.is_available():
                print("MPS/GPU is active")

            print("=====================")
            time.sleep(self.interval)
