import tensorflow as tf


class TextTransformations:
    @staticmethod
    def lower():
        return lambda msg: tf.strings.lower(msg)

    @staticmethod
    def map(fn):
        return lambda text: fn(text)

    @staticmethod
    def regexp_replace_multiple(text, kvps):
        for key, value in kvps:
            text = tf.strings.regex_replace(text, key, value)

        return text
