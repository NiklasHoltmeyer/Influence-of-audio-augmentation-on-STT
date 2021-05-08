import json
import os
import pandas as pd
import csv

from audioengine.model.finetuning.wav2vec2.preprocess.preprocess_dataset_settings import logger


class Text:
    @staticmethod
    def read_csv(*path, **kwargs):
        path = os.path.join(*path)

        if not os.path.exists(path):
            raise Exception(f"File not Found\n {path}")

        sep = kwargs.get("sep", ",")
        delimiter = kwargs.get("delimiter", None)
        header = kwargs.get("header", "infer")
        names = kwargs.get("names", None)
        index_col = kwargs.get("index_col", None)
        usecols = kwargs.get("usecols", None)
        squeeze = kwargs.get("squeeze", False)
        prefix = kwargs.get("prefix", None)
        mangle_dupe_cols = kwargs.get("mangle_dupe_cols", True)
        dtype = kwargs.get("dtype", None)
        engine = kwargs.get("engine", None)
        converters = kwargs.get("converters", None)
        true_values = kwargs.get("true_values", None)
        false_values = kwargs.get("false_values", None)
        skipinitialspace = kwargs.get("skipinitialspace", False)
        skiprows = kwargs.get("skiprows", None)
        skipfooter = kwargs.get("skipfooter", 0)
        nrows = kwargs.get("nrows", None)
        na_values = kwargs.get("na_values", None)
        keep_default_na = kwargs.get("keep_default_na", True)
        na_filter = kwargs.get("na_filter", True)
        verbose = kwargs.get("verbose", False)
        skip_blank_lines = kwargs.get("skip_blank_lines", True)
        parse_dates = kwargs.get("parse_dates", False)
        infer_datetime_format = kwargs.get("infer_datetime_format", False)
        keep_date_col = kwargs.get("keep_date_col", False)
        date_parser = kwargs.get("date_parser", None)
        dayfirst = kwargs.get("dayfirst", False)
        cache_dates = kwargs.get("cache_dates", True)
        iterator = kwargs.get("iterator", False)
        chunksize = kwargs.get("chunksize", None)
        compression = kwargs.get("compression", "infer")
        thousands = kwargs.get("thousands", None)
        decimal = kwargs.get("decimal", ".")
        lineterminator = kwargs.get("lineterminator", None)
        quotechar = kwargs.get("quotechar", '"')
        quoting = kwargs.get("quoting", csv.QUOTE_MINIMAL)
        doublequote = kwargs.get("doublequote", True)
        escapechar = kwargs.get("escapechar", None)
        comment = kwargs.get("comment", None)
        encoding = kwargs.get("encoding", None)
        dialect = kwargs.get("dialect", None)
        error_bad_lines = kwargs.get("error_bad_lines", True)
        warn_bad_lines = kwargs.get("warn_bad_lines", True)
        delim_whitespace = kwargs.get("delim_whitespace", False)
        memory_map = kwargs.get("memory_map", False)
        float_precision = kwargs.get("float_precision", None)
        storage_options = kwargs.get("storage_options", None)

        return pd.read_csv(path, sep=sep, delimiter=delimiter, header=header, names=names, index_col=index_col,
                           usecols=usecols, squeeze=squeeze, prefix=prefix, mangle_dupe_cols=mangle_dupe_cols,
                           dtype=dtype, engine=engine, converters=converters, true_values=true_values,
                           false_values=false_values, skipinitialspace=skipinitialspace, skiprows=skiprows,
                           skipfooter=skipfooter, nrows=nrows, na_values=na_values, keep_default_na=keep_default_na,
                           na_filter=na_filter, verbose=verbose, skip_blank_lines=skip_blank_lines,
                           parse_dates=parse_dates, infer_datetime_format=infer_datetime_format,
                           keep_date_col=keep_date_col, date_parser=date_parser, dayfirst=dayfirst,
                           cache_dates=cache_dates, iterator=iterator, chunksize=chunksize, compression=compression,
                           thousands=thousands, decimal=decimal, lineterminator=lineterminator, quotechar=quotechar,
                           quoting=quoting, doublequote=doublequote, escapechar=escapechar, comment=comment,
                           encoding=encoding, dialect=dialect, error_bad_lines=error_bad_lines,
                           warn_bad_lines=warn_bad_lines, delim_whitespace=delim_whitespace, memory_map=memory_map,
                           float_precision=float_precision, storage_options=storage_options)


def save_settings(path, settings, infos=None, indent=4, desc=""):
    """

    Args:
        path: Path
        settings: dict
        infos: Array of KVPs
    """

    infos = [] if not infos else infos

    for key, value in infos:
        settings[key] = value

    settings_json = json.dumps(settings, indent=indent)

    with open(path, "w") as f:
        f.write(settings_json)

    logger.debug(f"Saved {desc} Settings to {path}")