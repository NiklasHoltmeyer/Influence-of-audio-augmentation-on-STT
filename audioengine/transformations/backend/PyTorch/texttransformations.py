import re


class RegExp:
    def __init__(self, patterns):
        """

        Args:
            patterns: [(pattern, replacement), (pattern, replacement), ...]
        """
        self.patterns = patterns

    def __call__(self, data):
        for pattern, replacement in self.patterns:
            data["sentence"] = re.sub(pattern, replacement, data["sentence"])
        return data
