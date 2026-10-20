import time
import os
import psutil
import threading
from tokenizer import train_bpe, save_bpe, Tokenizer

train_path = r"../data/tinystories_train.txt"
eval_paths = [r"../data/tinystories_valid.txt", r"../data/owt_valid.txt"]

vocab_size = 10000
special_tokens = ["<|endoftext|>"]

def measure_peak_memory(func, *args):
    process = psutil.Process(os.getpid())
    peak_mem = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            try:
                mem = process.memory_info().rss
                peak_mem[0] = max(peak_mem[0], mem)
            except psutil.Error:
                pass
            stop.wait(0.01)

    t = threading.Thread(target=monitor)
    t.start()

    try:
        start = time.perf_counter()
        result = func(*args)
        spent_time = time.perf_counter() - start
    finally:
        stop.set()
        t.join()

    return result, spent_time, peak_mem[0]


print(f"\nTraining BPE on {train_path}...")
(vocab, merges), spent_time, peak_mem = measure_peak_memory(train_bpe, train_path, vocab_size, special_tokens)

longest_token = max(vocab.values(), key=len)

print(f"Vocab size: {len(vocab)}")
print(f"Merges size: {len(merges)}")
print(f"Peak memory: {peak_mem} bytes")
print(f"Training time: {spent_time} secs ({spent_time / 60} mins)")
print(f"Longest token: {longest_token!r} (length: {len(longest_token)} bytes)")
print(f"First 20 merges: {merges[:20]}")

#saving trained vocabs and merges
save_bpe(vocab, merges, "../data/vocab_train.json", "../data/merges_train.json")

#loading tokenizer for evaluation datasets
tokenizer = Tokenizer.from_files("../data/vocab_train.json", "../data/merges_train.json", special_tokens=special_tokens)

#evaluating on TinyStories validation and OWT validation
for input_path in eval_paths:
    print(f"\nEvaluating on {input_path}...")

    with open(input_path, "rb") as f:
        raw_text = f.read()

    text = raw_text.decode("utf-8")
    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    spent_time = time.perf_counter() - start

    byte_count = len(raw_text)
    token_count = len(tokens)

    print(f"Input size: {byte_count / 1_000_000} MB")
    print(f"Token count: {token_count}")
    print(f"Tokenization time: {spent_time} secs")
    print(f"Throughput: {byte_count / 1_000_000 / spent_time} MB/s")
    print(f"Bytes per token: {byte_count / token_count}")


#results:
# Training BPE on ../data/tinystories_train.txt...
# Vocab size: 10000
# Merges size: 9743
# Peak memory: 89767936 bytes
# Training time: 1251.0382787999697 secs (20.850637979999494 mins)
# Longest token: b' accomplishment' (length: 15 bytes)
# First 20 merges: [(b' ', b't'), (b'h', b'e'), (b' ', b'a'), (b' ', b's'), (b' ', b'w'), (b'n', b'd'), (b' t', b'he'), (b'e', b'd'), (b' ', b'b'), (b' t', b'o'), (b' a', b'nd'), (b' ', b'h'), (b' ', b'f'), (b'i', b'n'), (b' ', b'T'), (b' w', b'a'), (b'r', b'e'), (b'i', b't'), (b'o', b'u'), (b' ', b'l')]
# Evaluating on ../data/tinystories_valid.txt...
# Input size: 22.502601 MB
# Token count: 5465882
# Tokenization time: 82.96440749999601 secs
# Throughput: 0.27123198583683106 MB/s
# Bytes per token: 4.116920379913068
# Evaluating on ../data/owt_valid.txt...
# Input size: 289.998753 MB
# Token count: 91370967
# Tokenization time: 1198.1350417999784 secs
# Throughput: 0.24204179235449957 MB/s
# Bytes per token: 3.173861047131087