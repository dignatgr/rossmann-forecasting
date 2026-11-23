from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data(data_dir=DATA_DIR, only_open=True):
    """Load train and store tables, merge them and sort by store and date.

    If only_open is True, drop closed days and open days with zero sales.
    """
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
    store = pd.read_csv(data_dir / "store.csv")
    df = train.merge(store, on="Store", how="left")

    if only_open:
        df = df[(df["Open"] == 1) & (df["Sales"] > 0)]
    return df.sort_values(["Store", "Date"]).reset_index(drop=True)


def split_by_date(df, date):
    """ split the dataframe into train and validation sets based on a given date """
    train = df[df['Date'] < date]
    validation = df[df['Date'] >= date]
    return train, validation
