from queue import Queue
from threading import Thread
import time

# Object that signals shutdown
_sentinel = object()

compute_task = lambda a,b: (time.sleep(1), print("I am compute task"))
all_reduce_task = lambda x: (time.sleep(1.1), print("I am com task"))

# A thread that produces data
def producer(out_q):
    num_blocks = 10
    # Produce some data
    for x in range(num_blocks):
        compute_task(0, 0)
        out_q.put(all_reduce_task)

    # Put the sentinel on the queue to indicate completion
    out_q.put(_sentinel)

# A thread that consumes data
def consumer(in_q):
    while True:
# Get some data
        all_reduce_func = in_q.get()

        # Check for termination
        if all_reduce_func is _sentinel:
            in_q.put(_sentinel)
            break
        else:
            all_reduce_func(0)

# Create the shared queue and launch both threads
q = Queue()
t1 = Thread(target=consumer, args=(q,))
t2 = Thread(target=producer, args=(q,))
t1.start()
t2.start()
