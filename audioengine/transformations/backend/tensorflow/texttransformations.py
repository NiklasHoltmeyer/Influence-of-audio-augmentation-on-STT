import tensorflow as tf


class TextTransformations:
    @staticmethod
    def lower(x, y):
        """

        Args:
            x: inut
            y: output
        Returns:
            x, transform(y)
        """
        return x, tf.strings.lower(y)

    @staticmethod
    def map(fn):
        """

        Args:
            fn: Function-Pointer

        Returns:
            x, map(y)
        """
        return lambda x, y: fn(x, y)

    @staticmethod
    def regexp_replace_multiple(kvps):
        """

        Args:
            kvps: [(key, value), ...] Replace Key with Value

        Returns:
            __call__(x, y) function
            __call__(x, y) = x, transform(y)
        """
        def __call__(x, y):
            for key, value in kvps:
                y = tf.strings.regex_replace(y, key, value)
            return x, y
        return __call__
