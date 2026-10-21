import regex
import json
from collections import Counter, defaultdict
import base64

PAT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens: #adding the special tokens
        vocab[len(vocab)] = token.encode()

    #instead of reading the whole file at once, read it in chunks based on the special tokens and count pretokens in each chunk
    counts = count_pretokens_file(input_path, special_tokens)
    seqs = {w: tuple(bytes([b]) for b in w.encode()) for w in counts}

    #creating a dictionary of pairs appearing in each word
    pair_words_dict = defaultdict(set)
    pairs = Counter() #and their counts

    for w, c in counts.items():
        for a, b in zip(seqs[w], seqs[w][1:]):
            pairs[(a, b)] += c
            pair_words_dict[(a, b)].add(w) #mapping pairs to words

    merges = []
    while len(vocab) < vocab_size:
        if not pairs:
            break

        best = max(pairs, key=lambda p: (pairs[p], p))
        #words containing the selected pair
        affected_words = pair_words_dict[best].copy()

        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]

        #updating counts and pairs for affected words(otimizing by only updating words that contain the merged pair)
        for w in affected_words:
            c = counts[w]
            old_seq = seqs[w]
            old_pairs = list(zip(old_seq, old_seq[1:]))

            for pair in old_pairs: #removing old pair counts
                pairs[pair] -= c

            for pair in set(old_pairs): #removing the old pari/word dict entery
                pair_words_dict[pair].discard(w)

            #merging the best pair and updating sequence
            new_seq = merge(old_seq, best)
            seqs[w] = new_seq
            new_pairs = list(zip(new_seq, new_seq[1:]))

            for pair in new_pairs: #recounting the pairs
                pairs[pair] += c

            for pair in set(new_pairs): #readding the word to pair/word dict
                pair_words_dict[pair].add(w)

        for pair in list(pairs): #removing unused pairs
            if pairs[pair] <= 0:
                del pairs[pair]
                pair_words_dict.pop(pair, None)

    return vocab, merges

def count_pretokens(corpus, special_tokens):
    counts = Counter()
    if special_tokens:
        special_tokens_sorted = sorted(
            special_tokens, key=len, reverse=True
        )
        special_pattern = "|".join(
            regex.escape(token) for token in special_tokens_sorted
        )
        chunks = regex.split(f"({special_pattern})", corpus)
    else:
        chunks = [corpus]

    for chunk in chunks:
        if chunk in special_tokens:
            continue

        for token in regex.findall(PAT, chunk):
            counts[token] += 1

    return counts


def merge(seq, best): #merging the mpost frequent pair together to form a new token
    result = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
            result.append(seq[i] + seq[i + 1])
            i += 2
        else:
            result.append(seq[i])
            i += 1

    return tuple(result)


def save_bpe(vocab, merges, vocab_path, merges_path):
    with open(vocab_path, "w", encoding="utf-8") as f:
        #saving the vocabs as integer ids and base64 bytes
        json.dump({int(token_id): base64.b64encode(token).decode() for token_id, token in vocab.items()}, f, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        for left, right in merges:
            #saving merges as base64 bytes
            f.write(base64.b64encode(left).decode() + " " + base64.b64encode(right).decode() + "\n")

def count_pretokens_file(input_path, special_tokens):
    counts = Counter()
    if special_tokens is None:
        return counts

    with open(input_path, "r", encoding="utf-8") as f:
        lines = []

        for line in f:
            lines.append(line)

            if any(st in line for st in special_tokens):
                corpus = "".join(lines)
                counts.update(count_pretokens(corpus, special_tokens))
                lines = []

        if lines:
            corpus = "".join(lines)
            counts.update(count_pretokens(corpus, special_tokens))

    return counts


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab 
        self.merges = merges 
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.special = {token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab = {int(token_id): base64.b64decode(token_bytes) for token_id, token_bytes in raw_vocab.items()}
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2: #2 merged tokens
                    left, right = parts
                    merges.append((base64.b64decode(left), base64.b64decode(right)))

        return cls(vocab, merges, special_tokens)

    # def encode(self, text):
    #     ids = []
    #     for chunk in self.split_on_special_tokens(text, self.special_tokens):
    #         if chunk in self.special:
    #             ids.append(self.special[chunk])
    #             continue
    #         for pre in regex.findall(PAT, chunk):
    #             toks = [bytes([b]) for b in pre.encode()]
    #             for pair in self.merges:
    #                 toks = merge(toks, pair)
    #             ids.extend(self.token_to_id[t] for t in toks)
    #     return ids

    def encode(self, text):
        ids = []
        for chunk in self.split_on_special_tokens(text, self.special_tokens):
            # Special tokens are already complete tokens
            if chunk in self.special:
                ids.append(self.special[chunk])
                continue

            for pre in regex.findall(PAT, chunk):
                toks = [bytes([b]) for b in pre.encode()]

                #optimizing the loop instead of merging all pairs, merge the first pair
                while len(toks) > 1:
                    best_pair = None
                    best_rank = float("inf")

                    for i in range(len(toks) - 1):
                        pair = (toks[i], toks[i + 1])
                        rank = self.merge_ranks.get(pair)
                        if rank is not None and rank < best_rank:
                            best_pair = pair
                            best_rank = rank

                    if best_pair is None:
                        break
                    toks = merge(toks, best_pair)
                ids.extend(self.token_to_id[t] for t in toks)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    @staticmethod
    def split_on_special_tokens(text, special_tokens):
        if not special_tokens:
            return [text]
        special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(regex.escape(token) for token in special_tokens_sorted)
        return regex.split(f"({special_pattern})", text)


