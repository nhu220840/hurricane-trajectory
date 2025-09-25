import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv('data/ibtracs_track_ml.csv')
# profile = ProfileReport(data, title="Profiling Report")
# profile.to_file("hurricane.html")

print(data.info())