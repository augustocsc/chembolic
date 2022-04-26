import pandas as pd

def load_activation_energies():
    df = pd.read_csv("../data/activation_energies.csv", index_col=0)

    return df

def load_rate_constants():
    df = pd.read_csv("../data/rate_constants.csv", index_col=0)

    return df
