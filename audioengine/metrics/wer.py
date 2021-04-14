import wer
from datasets import load_metric
import jiwer


class Wer:
    def __init__(self, transformation=None):
        self.transformation = transformation
        self.wer = load_metric("wer")

    def add_batch(self, ground_truths, references):
        ground_truths = self.transformation(ground_truths) if self.transformation else ground_truths
        references = self.transformation(references) if self.transformation else references

        self.wer.add_batch(predictions=ground_truths, references=references)

    def calc(self):
        return self.wer.compute()


class Jiwer:
    def __init__(self, transformation=None):
        self.hits, self.substitutions, self.deletions, self.insertions = 0, 0, 0, 0
        self.sentences_compared = 0
        self.transformation = transformation

    def add_batch(self, ground_truths, references):
        for ground_truth, hypothesis in zip(ground_truths, references):
            self.add(ground_truth, hypothesis)

    def add(self, ground_truth, hypothesis):
        ground_truth = self.transformation(ground_truth) if self.transformation else ground_truth
        hypothesis = self.transformation(hypothesis) if self.transformation else hypothesis

        hits, substitutions, deletions, insertions = Jiwer.compute_measurements(ground_truth, hypothesis)
        self.hits += hits
        self.substitutions += substitutions
        self.deletions += deletions
        self.insertions += insertions
        self.sentences_compared += 1

    def calc(self):
        wer = float(self.substitutions + self.deletions + self.insertions) / \
              float(self.hits + self.substitutions + self.deletions)
        return wer

    @staticmethod
    def compute_measurements(ground_truth, measurements):
        compute_measures = jiwer.compute_measures(ground_truth, measurements)

        hits = compute_measures["hits"]
        substitutions = compute_measures["substitutions"]
        deletions = compute_measures["deletions"]
        insertions = compute_measures["insertions"]

        return hits, substitutions, deletions, insertions

if __name__ == "__main__":
    wer = Wer()

