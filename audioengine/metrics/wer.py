import jiwer
class WER:
    def __init__(self, transformation=None):
        self.hits, self.substitutions, self.deletions, self.insertions = 0, 0, 0, 0
        self.transformation = transformation

    def append(self, ground_truth, hypothesis):
        ground_truth = self.transformation(ground_truth) if self.transformation else ground_truth
        hypothesis = self.transformation(hypothesis) if self.transformation else hypothesis

        hits, substitutions, deletions, insertions = WER.compute_measurements(ground_truth, hypothesis)
        self.hits += hits
        self.substitutions += substitutions
        self.deletions += deletions
        self.insertions += insertions

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