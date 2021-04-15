from datasets import load_metric
from multiprocessing import Pool
import jiwer


class Jiwer:
    def __init__(self, transformation=None):
        self.hits, self.substitutions, self.deletions, self.insertions = 0, 0, 0, 0
        self.sentences_compared = 0
        self.transformation = transformation

    def add_batch(self, ground_truths, references, core_count=None):
        """

        Args:
            ground_truths: Ground Truths: list(str)
            references: Predictions: list(sr)
            core_count: None -> Single-Thread, n -> on `n` Threads

        Returns:
            None
        """
        jobs = zip(ground_truths, references)

        if core_count:
            for ground_truth, hypothesis in jobs:
                self.add(ground_truth, hypothesis)
        else:
            with Pool(core_count) as p:
                results = p.map(self.wer, jobs)
                for hits, substitutions, deletions, insertions in results:
                    self._add(hits, substitutions, deletions, insertions)

    def add(self, ground_truth, hypothesis):
        hits, substitutions, deletions, insertions = self.wer(ground_truth, hypothesis)
        self._add(hits, substitutions, deletions, insertions)

    def _add(self, hits, substitutions, deletions, insertions):
        self.hits += hits
        self.substitutions += substitutions
        self.deletions += deletions
        self.insertions += insertions
        self.sentences_compared += 1

    def wer(self, ground_truth, hypothesis):
        ground_truth = self.transformation(ground_truth) if self.transformation else ground_truth
        hypothesis = self.transformation(hypothesis) if self.transformation else hypothesis

        return Jiwer.compute_measurements(ground_truth, hypothesis)

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


if __name__ == "__main__":
    wer = Wer()