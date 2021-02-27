import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import find_peaks, peak_prominences
import os
from scipy.signal import butter,filtfilt

class LowpassFilter:
	def __init__(self, fps=0.2, order=2, cutoff=0.15):
		self.name = "butter filter"
		self.fs = 1/fps
		self.order = order
		self.cutoff = cutoff
		self.nyq = 0.5*self.fs

	def apply(self, data):
		normal_cutoff = self.cutoff / self.nyq
		# Get the filter coefficients 
		b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
		y = filtfilt(b, a, data)
		return y

class ReadData:
	def __init__(self, directory):
		self.data_norm, self.data_average = self.read(directory=directory)

	def deltanorm(self, cells):
	    labels = cells.columns.values.tolist()  
	    F0 = cells.iloc[0:10,:].describe().iloc[1,:].values
	    DeltaF = cells.values - F0
	    norm = DeltaF/F0
	    cells = pd.DataFrame(norm, columns = labels)
	    return cells

	def read(self, directory):
		data = pd.read_csv(directory)
		data_norm = self.deltanorm(cells=data)
		data_average = data_norm['Average']
		data_norm = data_norm.drop(columns=['Average','Err'])
		return data_norm, data_average
