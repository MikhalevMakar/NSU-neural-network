from collections import Counter
from math import log2
import pandas as pd


class GainRation:
    @staticmethod
    def _calculate_entropy(counter: Counter, total_rows: int) -> float:
        total_count = sum(counter.values())
        entropy = 0.0

        for value in counter.values():
            probability = value / total_count
            entropy -= probability * log2(probability)

        return entropy * total_count / total_rows

    @staticmethod
    def _inform_entropy_t(df: pd.DataFrame, target: str) -> float:
        counter = Counter(df[target])
        return GainRation._calculate_entropy(counter, df.shape[0])

    @staticmethod
    def _inform_entropy_a_t(df: pd.DataFrame, attribute: str, target: str) -> float:
        categories = df[attribute].unique()
        total_rows = df.shape[0]
        info_entropy = 0

        for category in categories:
            subset = df[df[attribute] == category]
            counter = Counter(subset[target])
            info_entropy += GainRation._calculate_entropy(counter, total_rows)

        return info_entropy

    @staticmethod
    def gain_ration(df: pd.DataFrame, attribute: str, target: str) -> float:
        inform_gain = GainRation._inform_entropy_t(df, target) - GainRation._inform_entropy_a_t(df, attribute, target)
        split_info = GainRation._inform_entropy_t(df, attribute)
        return inform_gain / split_info
