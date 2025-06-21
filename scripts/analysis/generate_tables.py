import pandas as pd

# initialise metric considered
metric = "gain_avg"
metric_rel = f"{metric}_rel"

# # this is for experiment A
# csv_path = "data/results_processed/1_2_3_4_5/metrics.csv"
# required_types = ("series_5", "series_4", "series_1", "series_3", "series_2")

# # this is for experiment B
# csv_path = "data/results_processed/1_6_7_8_9/metrics.csv"
# required_types = ("series_9", "series_8", "series_1", "series_7", "series_6")

# this is for experiment C
csv_path = "data/results_processed/1_10_11_12_13/metrics.csv"
required_types = ("series_13", "series_12", "series_1", "series_11", "series_10")

# read csv
df = pd.read_csv(csv_path, index_col=0)
print(df)

# filter df so that only records reflecting spreading regime common for all network_types are left
grouped = df.groupby(['protocol', 'probab', 'seed_budget', 'ss_method'])
filtered_df = grouped.filter(lambda g: set(g['net_type']) == set(required_types))
print(filtered_df)

# add a column with relative value of the analysed metric to the baseline series
reference = (
    filtered_df[filtered_df["net_type"] == "series_1"]
    .set_index(["protocol", "probab", "seed_budget", "ss_method"])[metric]
    .rename(f"metric_ref")
)
filtered_df = filtered_df.merge(
    reference,
    on=["protocol", "probab", "seed_budget", "ss_method"],
    how="left"
)
filtered_df[metric_rel] = (
    filtered_df[metric] - filtered_df["metric_ref"]
) / filtered_df["metric_ref"]
filtered_df = filtered_df.drop("metric_ref", axis=1)
print(filtered_df)

# # table grouped by protocol
# a = filtered_df.groupby(["protocol", "ss_method", "net_type"]).mean().reset_index()
# print(a)
# b = a[["protocol", "net_type", metric_rel]].pivot(index="protocol", columns="net_type")
# b = b[[(metric_rel, net_type) for net_type in required_types]]
# b = b.round(4)
# print(b)

# # table grouped by pi
# a = filtered_df.loc[filtered_df["protocol"] == "AND"].drop("protocol", axis=1)  # <- can be OR
# a = a.groupby(["probab", "ss_method", "net_type"]).mean().reset_index()
# print(a)
# b = a[["probab", "net_type", metric_rel]].pivot(index="probab", columns="net_type")
# b = b[[(metric_rel, net_type) for net_type in required_types]]
# b = b.round(4)
# print(b)

# # table grouped by s
# a = filtered_df.loc[filtered_df["protocol"] == "AND"].drop("protocol", axis=1)  # <- can be OR
# a = a.groupby(["seed_budget", "ss_method", "net_type"]).mean().reset_index()
# print(a)
# b = a[["seed_budget", "net_type", metric_rel]].pivot(index="seed_budget", columns="net_type")
# b = b[[(metric_rel, net_type) for net_type in required_types]]
# b = b.round(4)
# print(b)

# # this is for experiment A
# a = filtered_df
# for proto in a["protocol"].unique():
#     b = a.loc[a["protocol"] == proto]
#     for s in b["seed_budget"].unique():
#         c = b.loc[b["seed_budget"] == s]
#         print(proto, s)
#         d = c[["net_type", "probab", metric_rel]].pivot(index="probab", columns="net_type")
#         d = d[[(metric_rel, net_type) for net_type in required_types]]
#         d = d.round(4)
#         print(d)
#         d.to_latex(f"./A-{proto}-{s}.tex")

# # this is for experiment B
# for proto in ["AND", "OR"]:
#     a = filtered_df.loc[filtered_df["protocol"] == proto].drop("protocol", axis=1)
#     a = a.groupby(["probab", "ss_method", "net_type"]).mean().reset_index()
#     b = a[["probab", "net_type", metric_rel]].pivot(index="probab", columns="net_type")
#     b = b[[(metric_rel, net_type) for net_type in required_types]]
#     b = b.round(4)
#     print(b)
#     b.to_latex(f"./B-{proto}.tex")

# this is for experiment C
for proto in ["AND", "OR"]:
    a = filtered_df.loc[filtered_df["protocol"] == proto].drop("protocol", axis=1)
    a = a.groupby(["probab", "ss_method", "net_type"]).mean().reset_index()
    b = a[["probab", "net_type", metric_rel]].pivot(index="probab", columns="net_type")
    b = b[[(metric_rel, net_type) for net_type in required_types]]
    b = b.round(4)
    print(b)
    b.to_latex(f"./C-{proto}.tex")