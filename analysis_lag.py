import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from scipy.signal import find_peaks, peak_prominences
import os
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from phil_ducial_points import detrend


class LowpassFilter:
    def __init__(self, fps=0.2, order=2, cutoff=0.15):
        self.name = "butter filter"
        self.fs = 1 / fps
        self.order = order
        self.cutoff = cutoff
        self.nyq = 0.5 * self.fs

    def apply(self, data):
        normal_cutoff = self.cutoff / self.nyq
        # Get the filter coefficients
        b, a = butter(self.order, normal_cutoff, btype="low", analog=False)
        y = filtfilt(b, a, data)
        return y


class ReadData:
    def __init__(self, directory):
        self.data_norm, self.data_average = self.read(directory=directory)

    def deltanorm(self, cells):
        labels = cells.columns.values.tolist()
        F0 = cells.iloc[0:10, :].describe().iloc[1, :].values
        DeltaF = cells.values - F0
        norm = DeltaF / F0
        cells = pd.DataFrame(norm, columns=labels)
        return cells

    def read(self, directory):
        data = pd.read_csv(directory)
        data_norm = self.deltanorm(cells=data)
        data_average = data_norm["Average"]
        data_norm = data_norm.drop(columns=["Average", "Err"])
        return data_norm, data_average

class CellData:
    def __init__(self, er: ReadData, cy: ReadData, fps=0.859, frames=400):
        """
        Split all the cells in the dataframe into individual somethings.

        Args:
                fps: while the name is missleading, this is the time per frame in seconds.

        Returns:
                This is a description of what is returned.
        """

        # hier das warum beschreiben. Wo kommen die Zahlen her?
        time_both = np.arange(fps, frames * fps, fps)
        self.cells = []

        # each column is one cell
        for cell in range(len(er.data_norm.columns)):

            # load and process the er signal data
            er_signal = er.data_norm.iloc[:, cell].values
            filtering = LowpassFilter(fps)
            er_lowpass = filtering.apply(er_signal)

            # load and process the cytosol signal data
            # reuse the same filter.
            cy_signal = cy.data_norm.iloc[:, cell].values
            cy_lowpass = filtering.apply(cy_signal)

            # calculate the derivative of the signals
            er_diff = np.diff(er_lowpass) / np.diff(time_both)
            cy_diff = np.diff(cy_lowpass) / np.diff(time_both)

            self.cells.append(
                Cell(
                    Trace(er_signal, er_lowpass, er_diff),
                    Trace(cy_signal, cy_lowpass, cy_diff),
                    time_both,
                )
            )

        self.find_experiment_start(fps, frames)

        self.frequencies = []
        self.mean_lags = []
        for cell in self.cells:
            cell.get_oscillation_group(start_reaction=self.experiment_start_cy, after_start=self.experiment_start_cy+200)
            cell.get_frequency(fps)
            self.frequencies.append(cell.frequency_per_min)
            cell.get_mean_peak_lag()
            self.mean_lags.append(cell.mean_lag)
            

    def find_experiment_start(self, fps=0.859, frames=400):
        self.avg_signal_er = np.mean([cell.er.signal for cell in self.cells], axis=0)
        self.avg_signal_cy = np.mean([cell.cy.signal for cell in self.cells], axis=0)

        # the onset of the experiment should be found if we find the fiducial point between beginning and maximum value
        # we can calculate this for both, er and cy
        point_a = 0
        point_b = np.argmax(self.avg_signal_cy)
        self.experiment_start_cy = np.argmin(detrend(self.avg_signal_cy[point_a:point_b]))

        point_a = 100
        point_b = np.argmin(self.avg_signal_er)
        self.experiment_start_er = np.argmax(detrend(self.avg_signal_er[point_a:point_b]))+100
        self.time_both = np.arange(fps, frames * fps, fps)




@dataclass
class Trace:
    signal: np.ndarray
    lowpass: np.ndarray
    derivative: np.ndarray


@dataclass
class TurningPoints:
    matches: np.ndarray
    lag: np.ndarray


@dataclass
class Cell:
    er: Trace
    cy: Trace
    time: np.ndarray

    @staticmethod
    def find_peaks(signal, prominence=0.01, flip=False):
        if flip:
            signal = -signal

        return find_peaks(signal, prominence=prominence)[0]

    def calculate_lag(self, signal_a, signal_b, flip_a=False, flip_b=False):
        signal_a_peaks = self.find_peaks(signal_a, flip=flip_a, prominence=0.04)
        signal_b_peaks = self.find_peaks(signal_b, flip=flip_b, prominence=0.05)

        last_index = 0
        matches = []

        # match every er peak to every cy peak that is before that one.
        for end in signal_a_peaks:
            # find closest match
            # we want the largest smaller value
            candidates = signal_b_peaks[last_index::][
                signal_b_peaks[last_index::] < end
            ]

            if len(candidates):
                start = candidates.argmax()

                matches.append([signal_b_peaks[last_index + start], end])
                last_index += start + 1

        matches = np.array(matches)
        lag = self.time[matches[:, 1]] - self.time[matches[:, 0]]

        return matches, lag

    def calculate_lag_flavors(self):
        # cy up, er down
        matches, lag = self.calculate_lag(
            self.er.derivative, self.cy.derivative, flip_a=True
        )
        self.cy_influx = TurningPoints(matches, lag)

        matches, lag = self.calculate_lag(
            self.er.derivative, self.cy.derivative, flip_b=True
        )
        self.er_influx = TurningPoints(matches, lag)

        matches, lag = self.calculate_lag(self.er.lowpass, self.cy.lowpass, flip_a=True)
        self.cy_peak = TurningPoints(matches, lag)

        matches, lag = self.calculate_lag(self.er.lowpass, self.cy.lowpass, flip_b=True)
        self.er_peak = TurningPoints(matches, lag)


    def get_oscillation_group(self, start_reaction=None, after_start=350):
        self.variance_er = pd.Series(self.er.lowpass).rolling(20).var()
        self.variance_cy = pd.Series(self.cy.lowpass).rolling(20).var()
    
        variance_cy_sum = self.variance_cy.cumsum()	
        variance_cy_sum_end = self.variance_cy[after_start::].cumsum()

        if variance_cy_sum[start_reaction]>1 and variance_cy_sum_end.max()>0.5:
            self.oscillation = 'before_during_after'
        elif variance_cy_sum[start_reaction]>1:
            self.oscillation = 'before_during'
        elif variance_cy_sum_end.max()>0.5: 
            self.oscillation = 'during_after'
        else:
            self.oscillation = 'during'	

    def get_frequency(self, fps, start=0, end=400):
        peaks_cy, _ = find_peaks(self.cy.signal[start:end], prominence=0.5)
        #print(peaks_cy)
        if len(peaks_cy)>2:
            diffs = np.diff(peaks_cy)
            diffs = np.array([diff for diff in diffs if diff>5])
            true_oscillations = [i < np.min(diffs)*2.5 for i in diffs]
            mean_oscillations = np.mean(diffs[true_oscillations])
            time_per_period = mean_oscillations * fps
            self.frequency_per_min = 1/(time_per_period/60)      
        else:
            self.frequency_per_min = None

    def get_mean_peak_lag(self):
        try:
            self.calculate_lag_flavors()
            high_var = np.where(self.variance_er>0.0015)[0]
            true_matches = []
            true_lags = []
            for match, lag in zip(self.cy_peak.matches, self.cy_peak.lag):        
                if match[1] in high_var:
                    true_matches.append(match)
                    true_lags.append(lag)
                    
            lags = []
            if len(true_matches) > 1:
                for lag in true_lags:
                    lags.append(lag) 
                self.mean_lag = np.mean(lags)
            else:
                self.mean_lag = None
        except:
            self.mean_lag = None
            pass






