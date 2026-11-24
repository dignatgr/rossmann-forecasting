import pandas as pd
import numpy as np
from datetime import date

MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

def add_promo2_active(df):
    """Add Promo2Active: 1 if the store has joined Promo2 and the row's month
    is one of its launch months (PromoInterval), else 0."""
    df = df.copy()
    n_rows = len(df)
    stores = df[["Store", "Promo2SinceYear", "Promo2SinceWeek", "PromoInterval"]].drop_duplicates("Store").copy()

    # Monday of the ISO week when the store joined Promo2 (NaT if it never did)
    stores["Promo2Start"] = pd.to_datetime(stores.apply(
        lambda r: pd.NaT if pd.isna(r["Promo2SinceYear"])
        else date.fromisocalendar(int(r["Promo2SinceYear"]), int(r["Promo2SinceWeek"]), 1),
        axis=1))

    # one 0/1 column per calendar month: is it a launch month for this store?
    interval = stores["PromoInterval"].fillna("")
    month_cols = [f"m{m}" for m in MONTH_ABBR]
    for m, abbr in MONTH_ABBR.items():
        stores[f"m{m}"] = interval.str.contains(abbr).astype(int)   # "Sep" also matches "Sept"

    df = df.merge(stores[["Store", "Promo2Start"] + month_cols], on="Store", how="left")
    assert len(df) == n_rows, "merge changed the number of rows"

    # pick, for each row, the flag of its own month
    row_month = df["Date"].dt.month.to_numpy()
    is_launch_month = df[month_cols].to_numpy()[np.arange(len(df)), row_month - 1]

    # comparisons with NaT are False, so stores without Promo2 get 0
    df["Promo2Active"] = ((df["Date"] >= df["Promo2Start"]) & (is_launch_month == 1)).astype(int)
    return df.drop(columns=month_cols + ["Promo2Start"])["Promo2Active"]

def get_holiday_dates(raw):
    """Sorted array of dates flagged as holidays (use the unfiltered data, closed days included)."""
    return np.sort(raw.loc[raw["StateHoliday"] != "0", "Date"].unique())

def add_calendar_features(df):
    """Add calendar features derived from Date."""
    df = df.copy()
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    return df

def add_features(df, holiday_dates, cap=15):
    """Add calendar, competition, Promo2 and holiday features, and cast categorical columns.

    holiday_dates: sorted array of holiday dates (see get_holiday_dates).
    cap: max value (days) for DaysToHoliday / DaysSinceHoliday.
    """

    # calendar
    df = add_calendar_features(df)

    # competition
    df["CompetitionMonthsOpen"] = (
        12 * (df["Date"].dt.year - df["CompetitionOpenSinceYear"])
        + (df["Date"].dt.month - df["CompetitionOpenSinceMonth"])
    ).clip(lower=0)

    # Promo2 active
    df["Promo2Active"] = add_promo2_active(df).to_numpy()

    # Holiday dates
    hol = holiday_dates.astype("datetime64[D]")
    d = df["Date"].to_numpy().astype("datetime64[D]")

    nxt = np.searchsorted(hol, d, side="left")         # first holiday on or after the date
    prv = np.searchsorted(hol, d, side="right") - 1    # last holiday on or before the date

    days_to = np.where(nxt < len(hol), (hol[np.minimum(nxt, len(hol) - 1)] - d).astype(int), cap)
    days_since = np.where(prv >= 0, (d - hol[np.maximum(prv, 0)]).astype(int), cap)

    df["DaysToHoliday"] = np.minimum(days_to, cap)
    df["DaysSinceHoliday"] = np.minimum(days_since, cap)

    # convert
    cat_cols = ["StoreType", "Assortment", "StateHoliday", "Store"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    return df
