from textwrap import wrap
import numpy as np


class SimpleCharEmbedding:
    def __init__(self, max_len=50):
        self.unknown_key = "#"
        self.sos = "<"
        self.eos = ">"
        self.vocab = (
                [self.unknown_key, self.sos, self.eos, " "] +
                [chr(i + 96) for i in range(1, 27)] +
                list(".,?!öäüß-")
        )

        self.max_len = max_len
        self.char_to_idx = {}
        for idx, char in enumerate(self.vocab):
            self.char_to_idx[char] = idx

    def __call__(self, text):
        text = text.lower()
        text = self.sos + text + self.eos

        idxs = [self.char_to_idx.get(ch, 1) for ch in text]  # [[char -> idx]]

        idxs_chunked = chunks(idxs, self.max_len)

        last_len = len(idxs_chunked[-1])
        padding = self.max_len - last_len

        idxs_chunked[-1] = idxs_chunked[-1] + [0] * padding

        return idxs_chunked

    def inverse(self, idxs):
        idxs = np.array(idxs).flatten()
        return "".join(self.vocab[idx] for idx in idxs)


def chunks(l, n): return [l[x: x + n] for x in range(0, len(l), n)]


if __name__ == "__main__":
    embedding = SimpleCharEmbedding(max_len=10)

    text = "abcdefghijklmnopqrstuvwxyz"
    idxs = embedding(text)
    for idx in idxs:
        print(len(idx), idx)
    text_decode = embedding.inverse(idxs)
    print(text_decode)
    print(text in text_decode)

    # for x in embedding("abcdefghijklmnopqrstuvwxyz"):
    # print(len(x), x)
