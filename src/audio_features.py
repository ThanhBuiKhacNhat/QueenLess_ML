import numpy as np


class AudioFeatures:
    def __init__(self, s_mean: np.ndarray, s_std: np.ndarray, s_merged: np.ndarray,
                 d_mean: np.ndarray, d_std: np.ndarray, d_merged: np.ndarray) -> None:
        self.__s_mean = s_mean
        self.__s_std = s_std
        self.__s_merged = s_merged
        self.__d_mean = d_mean
        self.__d_std = d_std
        self.__d_merged = d_merged

    @property
    def s_mean(self) -> np.ndarray:
        return self.__s_mean

    @property
    def s_std(self) -> np.ndarray:
        return self.__s_std

    @property
    def s_merged(self) -> np.ndarray:
        return self.__s_merged

    @property
    def d_mean(self) -> np.ndarray:
        return self.__d_mean

    @property
    def d_std(self) -> np.ndarray:
        return self.__d_std

    @property
    def d_merged(self) -> np.ndarray:
        return self.__d_merged
