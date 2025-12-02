import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

import statsmodels.api as sm






def _prettify_axes(x_rotation=0):

    ax = plt.gca()

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=x_rotation, ha="right" if x_rotation else "center")






def overview_and_clean(df):

    if "date_event" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date_event"]):

        df["date_event"] = pd.to_datetime(df["date_event"], errors="coerce")


    required_cols = [

        "sector",

        "symbol",

        "date_event",

        "eps",

        "estimatedEPS",

        "eps_surprise",

        "PRE_CAR",

        "AR_0",

        "AR_1",

        "DRIFT_2_5",

        "DRIFT_1_5",

        "sentiment_0_10",

    ]

    required_cols = [c for c in required_cols if c in df.columns]


    df = df.dropna(subset=required_cols, how="any")

    return df




def add_surprise_group(df):

    if "eps_surprise" not in df.columns:

        df["surprise_group"] = "missing"

        return df


    big_threshold = 0.10

    small_threshold = 0.01


    def classify_surprise(x):

        if pd.isna(x):

            return "missing"

        elif x >= big_threshold:

            return "big beat"

        elif x >= small_threshold:

            return "beat"

        elif x <= -big_threshold:

            return "big miss"

        elif x <= -small_threshold:

            return "miss"

        else:

            return "meet"


    df["surprise_group"] = df["eps_surprise"].apply(classify_surprise)

    return df




def add_percent_eps_surprise(df):

    if "eps_surprise" not in df.columns or "estimatedEPS" not in df.columns:

        df["eps_surprise_pct"] = None

        return df


    denom = df["estimatedEPS"].abs().replace(0, pd.NA)

    df["eps_surprise_pct"] = df["eps_surprise"] / denom

    return df




def add_sentiment_bucket(df):


    if "sentiment_0_10" not in df.columns:

        df["sentiment_bucket"] = "missing"

        return df


    def bucket_sentiment(s):

        if pd.isna(s):

            return "missing"

        elif s < 5:

            return "bad"

        elif s <= 7:

            return "neutral"

        else:

            return "good"


    df["sentiment_bucket"] = df["sentiment_0_10"].apply(bucket_sentiment)

    return df




def add_time_features(df):

    if "date_event" in df.columns:

        df["event_year"] = df["date_event"].dt.year

    return df




def statistical_highlights(df):


    if "surprise_group" not in df.columns or "AR_0" not in df.columns:

        print("Required columns missing; skipping tests.\n")

        return


    big_beat = df[df["surprise_group"] == "big beat"]["AR_0"].dropna()

    big_miss = df[df["surprise_group"] == "big miss"]["AR_0"].dropna()


    if len(big_beat) < 2 or len(big_miss) < 2:

        print("Not enough observations in big beat / big miss groups for tests.\n")

        return


    t_bb, p_bb = stats.ttest_1samp(big_beat, 0.0)

    t_bm, p_bm = stats.ttest_1samp(big_miss, 0.0)

    t_diff, p_diff = stats.ttest_ind(big_beat, big_miss, equal_var=False)


    print(f"Big beat  : n = {len(big_beat):3d}, "

          f"mean AR_0 = {big_beat.mean(): .4f}, t = {t_bb: .2f}, p = {p_bb:.4f}")

    print(f"Big miss  : n = {len(big_miss):3d}, "

          f"mean AR_0 = {big_miss.mean(): .4f}, t = {t_bm: .2f}, p = {p_bm:.4f}")

    print(f"Difference (big beat - big miss): t = {t_diff: .2f}, p = {p_diff:.4f}\n")




def run_regression_models(df):

    needed = {"AR_0", "eps_surprise", "sentiment_0_10", "PRE_CAR"}

    if not needed.issubset(df.columns):

        return


    reg_df = df[list(needed)].dropna()

    if len(reg_df) < 10:

        return


    y = reg_df["AR_0"]

    X = reg_df[["eps_surprise", "sentiment_0_10", "PRE_CAR"]]

    X = sm.add_constant(X)


    model = sm.OLS(y, X).fit()


    print(f"Sample size used: n = {len(reg_df)}")

    print(f"R-squared: {model.rsquared:.3f}, "

          f"Adj. R-squared: {model.rsquared_adj:.3f}")

    print(f"F-test p-value (overall model): {model.f_pvalue:.4f}\n")


    for name in model.params.index:

        coef = model.params[name]

        pval = model.pvalues[name]

        print(f"{name:>12}: coef = {coef: .4f}, p-value = {pval:.4f}")

    print()




def plot_mean_AR0_by_surprise(df):

    if "surprise_group" not in df.columns or "AR_0" not in df.columns:

        return


    summary = df.groupby("surprise_group")["AR_0"].mean()


    plt.figure(figsize=(8, 5))

    summary.plot(kind="bar")

    plt.title("Mean AR_0 by EPS Surprise Group")

    plt.xlabel("EPS Surprise Group")

    plt.ylabel("Mean AR_0")

    plt.axhline(0, color="black")

    _prettify_axes(x_rotation=30)

    plt.tight_layout()

    plt.show()




def plot_event_counts_by_surprise(df):

    if "surprise_group" not in df.columns:

        return


    counts = df["surprise_group"].value_counts().sort_index()


    plt.figure(figsize=(8, 5))

    counts.plot(kind="bar")

    plt.title("Number of Events by EPS Surprise Group")

    plt.xlabel("EPS Surprise Group")

    plt.ylabel("Count of Events")

    plt.axhline(0, color="black")

    _prettify_axes(x_rotation=30)

    plt.tight_layout()

    plt.show()




def plot_mean_AR0_by_year(df):

    if "event_year" not in df.columns or "AR_0" not in df.columns:

        return


    summary = df.groupby("event_year")["AR_0"].mean()


    plt.figure(figsize=(8, 5))

    summary.plot(kind="bar")

    plt.title("Mean AR_0 by Event Year")

    plt.xlabel("Event Year")

    plt.ylabel("Mean AR_0")

    plt.axhline(0, color="black")

    _prettify_axes()

    plt.tight_layout()

    plt.show()




def main():

    df = pd.read_csv("FINAL_DATA.csv")


    num_cols = [

        "eps", "estimatedEPS", "eps_surprise",

        "PRE_CAR", "AR_0", "AR_1",

        "DRIFT_2_5", "DRIFT_1_5",

        "sentiment_0_10"

    ]

    for c in num_cols:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce")


    df["beat_miss"] = np.where(df["eps_surprise"] > 0, "Beat", "Miss")


    df = overview_and_clean(df)


    df = add_surprise_group(df)

    df = add_percent_eps_surprise(df)

    df = add_sentiment_bucket(df)

    df = add_time_features(df)




    df_prof = df.dropna(subset=["PRE_CAR", "AR_0", "AR_1", "DRIFT_1_5"]).copy()


    df_prof["CAR_-5"] = 0.0

    df_prof["CAR_-1"] = df_prof["PRE_CAR"]

    df_prof["CAR_0"]  = df_prof["PRE_CAR"] + df_prof["AR_0"]

    df_prof["CAR_1"]  = df_prof["PRE_CAR"] + df_prof["AR_0"] + df_prof["AR_1"]

    df_prof["CAR_5"]  = df_prof["PRE_CAR"] + df_prof["AR_0"] + df_prof["DRIFT_1_5"]


    car_cols = ["CAR_-5", "CAR_-1", "CAR_0", "CAR_1", "CAR_5"]

    taus = [-5, -1, 0, 1, 5]

    sectors = ["Tech", "Financials", "HealthCare"]


    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)


    for ax, sec in zip(axes, sectors):

        sub = df_prof[df_prof["sector"] == sec]

        if sub.empty:

            ax.set_visible(False)

            continue


        for outcome, color in zip(["Beat", "Miss"], ["tab:blue", "tab:orange"]):

            g = sub[sub["beat_miss"] == outcome]

            if g.empty:

                continue

            mean_cars = g[car_cols].mean() * 100  

            ax.plot(

                taus,

                mean_cars.values,

                marker="o",

                label=f"{outcome} (n={len(g)})",

                color=color,

            )


        ax.axhline(0, linestyle="--", linewidth=1, color="gray")

        ax.axvline(0, linestyle=":", linewidth=1, color="red")

        ax.set_title(sec)

        ax.set_xlabel("Earnings Timeline (days)")

        ax.grid(alpha=0.3)


    axes[0].set_ylabel("Average Abnormal Return (%)")


    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, title="Earnings Outcome",

               loc="lower left", bbox_to_anchor=(0.02, 0.02))

    fig.suptitle("Earnings Reaction Profiles by Sector\n(Beats vs Misses, -5 to +5 days)", y=1.02)

    plt.tight_layout()

    plt.show()


    df_pre = df.dropna(subset=["PRE_CAR"]).copy()

    sectors_order = ["Financials", "Tech", "HealthCare"]


    x = np.arange(len(sectors_order))

    width = 0.35


    means_beat = []

    means_miss = []

    n_beat = []

    n_miss = []


    for sec in sectors_order:

        sub = df_pre[df_pre["sector"] == sec]


        b = sub[sub["beat_miss"] == "Beat"]["PRE_CAR"]

        m = sub[sub["beat_miss"] == "Miss"]["PRE_CAR"]


        means_beat.append(b.mean() * 100 if len(b) > 0 else np.nan)

        means_miss.append(m.mean() * 100 if len(m) > 0 else np.nan)

        n_beat.append(len(b))

        n_miss.append(len(m))


    plt.figure(figsize=(8, 5))

    bars_beat = plt.bar(x - width/2, means_beat, width, label="Beat", color="tab:blue")

    bars_miss = plt.bar(x + width/2, means_miss, width, label="Miss", color="tab:orange")


    plt.axhline(0, linestyle="--", color="gray", linewidth=1)


    for i, (bb, bm) in enumerate(zip(bars_beat, bars_miss)):

        if not np.isnan(means_beat[i]):

            plt.text(bb.get_x() + bb.get_width()/2, bb.get_height() + 0.05,

                     f"n={n_beat[i]}", ha="center", va="bottom", fontsize=9)

        if not np.isnan(means_miss[i]):

            plt.text(bm.get_x() + bm.get_width()/2, bm.get_height() - 0.35,

                     f"n={n_miss[i]}", ha="center", va="top", fontsize=9, color="black")


    plt.xticks(x, sectors_order)

    plt.ylabel("Pre-earnings abnormal return (%)")

    plt.title("Pre-Earnings Abnormal Return by Sector and Earnings Outcome")

    plt.legend(title="Earnings Outcome", loc="lower right")

    plt.tight_layout()

    plt.show()


    df_day0 = df[df["sector"] == "Tech"].copy()

    df_day0 = df_day0.dropna(subset=["AR_0", "eps_surprise", "sentiment_0_10"])


    big_surp = 0.20

    pos_sent = 6.0


    def surprise_size(row):

        return "Big" if abs(row["eps_surprise"]) >= big_surp else "Small"


    def tone_bucket(row):

        return "Positive tone" if row["sentiment_0_10"] >= pos_sent else "Neutral tone"


    df_day0["surprise_size"] = df_day0.apply(surprise_size, axis=1)

    df_day0["tone_bucket"] = df_day0.apply(tone_bucket, axis=1)

    df_day0["bucket"] = df_day0["surprise_size"] + " Surprise, " + df_day0["tone_bucket"]


    bucket_order = [

        "Big Surprise, Positive tone",

        "Big Surprise, Neutral tone",

        "Small Surprise, Positive tone",

        "Small Surprise, Neutral tone",

    ]

    means = (

        df_day0.groupby("bucket")["AR_0"]

        .mean()

        .reindex(bucket_order)

    ) * 100


    plt.figure(figsize=(8, 4))

    plt.bar(bucket_order, means.values)

    plt.axhline(0, linestyle="--", color="gray", linewidth=1)

    plt.ylabel("Abnormal Return on day 0 (%)")

    plt.title("Abnormal Returns by Surprise and Sentiment (Tech)")

    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()

    plt.show()


    statistical_highlights(df)

    run_regression_models(df)


    plot_mean_AR0_by_surprise(df)

    plot_event_counts_by_surprise(df)

    plot_mean_AR0_by_year(df)




if __name__ == "__main__":

    main()
