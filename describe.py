import numpy as np
import pandas as pd
import argparse


class Describer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.columns = dataset.columns.values
        self.header = []
        self.dict = {
            "count": [],
            "mean": [],
            "std": [],
            "min": [],
            "25%": [],
            "50%": [],
            "75%": [],
            "max": [],
        }
        self.means = {}

    def ft_count(self, column):
        c = 0.0
        for col in self.dataset[column]:
            if not np.isnan(col):
                c = c + 1
        return c

    def ft_mean(self, column):
        sum = 0.0
        c = 0.0
        for col in self.dataset[column]:
            if not np.isnan(col):
                c = c + 1
                sum = sum + col
        return sum / c

    def ft_std(self, column):
        mean = self.ft_mean(column)
        somme = 0.0
        c = 0.0
        for x in self.dataset[column]:
            if not np.isnan(x):
                somme = somme + ((x - mean) ** 2)
                c = c + 1.0
        return np.sqrt(somme / (c - 1))

    def ft_min(self, column):
        val = self.dataset[column][0]
        for x in self.dataset[column]:
            if not np.isnan(x):
                if val > x:
                    val = x
        return val

    def ft_max(self, column):
        val = self.dataset[column][0]
        for x in self.dataset[column]:
            if not np.isnan(x):
                if val < x:
                    val = x
        return val

    # To calculate an interpolated percentile, do the following:

    # Calculate the rank to use for the percentile. Use: rank = p(n+1), where p = the percentile and n = the sample size.
    # For our example, to find the rank for the 70th percentile, we take 0.7*(11 + 1) = 8.4.

    # If the rank in step 1 is an integer, find the data value that corresponds to that rank and use it for the percentile.

    # If the rank is not an integer, you need to interpolate between the two closest observations.
    # For our example, 8.4 falls between 8 and 9, which corresponds to the data values of 35 and 40.

    # Take the difference between these two observations and multiply it by the fractional portion of the rank. For our example, this is: (40 – 35)0.4 = 2.

    # Take the lower-ranked value in step 3 and add the value from step 4 to obtain the interpolated value for the percentile. For our example, that value is 35 + 2 = 37.

    def ft_percentiles(self, column, perc):
        df = sorted(self.dataset[column].dropna())
        n = len(df)
        rank = (perc / 100) * ((n - 1))
        f = np.floor(rank)
        c = np.ceil(rank)
        if f == c:
            return df[int(rank)]
        v2 = df[int(f)]
        v3 = df[int(c)]
        rank = v2 + (rank - f) * (v3 - v2)
        return rank

    def describe(self):
        for col in self.columns:
            if np.issubdtype(self.dataset[col].dtype, np.number):
                self.header.append(col)
                self.dict["count"].append(self.ft_count(col))
                self.dict["mean"].append(self.ft_mean(col))
                self.dict["std"].append(self.ft_std(col))
                self.dict["min"].append(self.ft_min(col))
                self.dict["25%"].append(self.ft_percentiles(col, 25))
                self.dict["50%"].append(self.ft_percentiles(col, 50))
                self.dict["75%"].append(self.ft_percentiles(col, 75))
                self.dict["max"].append(self.ft_max(col))
        pd.options.display.float_format = '{:.4f}'.format
        return pd.DataFrame(self.dict, index=self.header).T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Path to .csv file")
    args = parser.parse_args()

    try:
        dataset = pd.read_csv(args.filename, header=None)
        dataset = Describer(dataset)
        print(dataset.describe())
    except Exception:
        print("Can't open file.")
