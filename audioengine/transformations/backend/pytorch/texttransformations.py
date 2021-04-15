import re


class Regexp:
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


class ToLower:
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        data[self.key] = data[self.key].lower()
        return data


if __name__ == "__main__":
    data = {
        "sentence": "aSsSSdsdsSD"
    }
    print(ToLower("sentence")(data))
