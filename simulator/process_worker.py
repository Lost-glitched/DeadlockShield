"""
Worker process — runs in its own OS process (multiprocessing.Process).

This is the IPC producer side of the architecture:

    Worker Processes  --(multiprocessing.Queue)-->  Resource Manager

Each worker represents one logical "process" from the workload. It sends
its resource-request chunks to the central Resource Manager over a shared
request queue, and waits for a decision on its own private response queue.
If a request is DEFERRED or TIMES OUT, the worker retries with a small
bounded backoff (this models a process that keeps trying rather than
giving up immediately) up to MAX_ATTEMPTS, after which it gives up on
that chunk and moves on (recorded as a permanently-failed request).
"""

import time

MAX_ATTEMPTS = 5
RETRY_BACKOFF_SEC = 0.01


def worker_process(pid, chunks, request_queue, response_queue, thread_slot):
    for chunk in chunks:
        if all(v == 0 for v in chunk):
            continue
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            send_time = time.perf_counter()
            request_queue.put(
                {
                    "pid": pid,
                    "type": "REQUEST",
                    "vector": chunk,
                    "send_time": send_time,
                    "thread_slot": thread_slot,
                }
            )
            response = response_queue.get()  # blocks until manager replies
            if response["decision"] in ("GRANTED",):
                break
            attempts += 1
            time.sleep(RETRY_BACKOFF_SEC)
        # loop ends either GRANTED or attempts exhausted -> chunk abandoned

    request_queue.put({"pid": pid, "type": "RELEASE", "thread_slot": thread_slot})
