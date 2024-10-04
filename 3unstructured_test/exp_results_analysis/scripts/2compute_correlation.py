import numpy as np
import scipy.stats as stats
import pandas as pd
import os


def transform_statistic_spear(x, y):
    rs = stats.spearmanr(x, y).statistic
    if rs == 1:
        transformed = 10000000000000
    else:
        transformed = 0.5 * np.log((1 + rs) / (1 - rs))
    return transformed


def fisher_pvalue_spear(x, y):
    result = stats.permutation_test(
        (x, y),
        transform_statistic_spear,
        alternative='two-sided',
        permutation_type='pairings')
    return result.pvalue


def transform_statistic_pear(x, y):
    rs = stats.pearsonr(x, y).statistic
    if rs == 1:
        transformed = 10000000000000
    else:
        transformed = 0.5 * np.log((1 + rs) / (1 - rs))
    return transformed


def fisher_pvalue_pear(x, y):
    result = stats.permutation_test(
        (x, y),
        transform_statistic_pear,
        alternative='two-sided',
        permutation_type='pairings')
    return result.pvalue


def main1(posterior_fp, output_dir):
    df = pd.read_csv(posterior_fp)
    list_outs = []
    for model_name in df["model_name"].unique():
        df2 = df[df["model_name"] == model_name]
        list1 = []
        list2 = []
        for field in ["A", "B", "C"]:
            list1.extend(df2[f"p({field})"].dropna().tolist())
            list2.extend(df2[f"p({field}|E)"].dropna().tolist())

        print(f"{model_name}: prior v.s. posterior, Spearman")
        spearman = stats.spearmanr(list1, list2).statistic
        spearman_pvalue = fisher_pvalue_spear(list1, list2)

        print(f"{model_name}: prior v.s. posterior, Pearson")
        pearson = stats.pearsonr(list1, list2).statistic
        pearson_pvalue = fisher_pvalue_pear(list1, list2)

        out = {"model_name": model_name,
               "spearman": spearman,
               "spearman_pvalue": spearman_pvalue,
               "pearson": pearson,
               "pearson_pvalue": pearson_pvalue}
        list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output_fp = os.path.join(output_dir, "1prior_posterior.csv")
    out_df.to_csv(output_fp, index=False)


def main2(posterior_fp, output_dir):
    df = pd.read_csv(posterior_fp)
    list_outs = []
    for model_name in df["model_name"].unique():
        df2 = df[df["model_name"] == model_name]
        list1 = []
        list2 = []
        for field in ["A", "B", "C"]:
            list1.extend(df2[f"sim_{field}"].dropna().tolist())
            list2.extend(df2[f"p({field}|E)"].dropna().tolist())

        print(f"{model_name}: similarity v.s. posterior, Spearman")
        spearman = stats.spearmanr(list1, list2).statistic
        spearman_pvalue = fisher_pvalue_spear(list1, list2)

        print(f"{model_name}: similarity v.s. posterior, Pearson")
        pearson = stats.pearsonr(list1, list2).statistic
        pearson_pvalue = fisher_pvalue_pear(list1, list2)

        out = {"model_name": model_name,
               "spearman": spearman,
               "spearman_pvalue": spearman_pvalue,
               "pearson": pearson,
               "pearson_pvalue": pearson_pvalue}
        list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output_fp = os.path.join(output_dir, "2similarity_posterior.csv")
    out_df.to_csv(output_fp, index=False)


def main3(posterior_fp, output_dir):
    df = pd.read_csv(posterior_fp)
    list_outs = []
    for model_name in df["model_name"].unique():
        df2 = df[df["model_name"] == model_name]
        list1 = []
        list2 = []
        for field in ["A", "B", "C"]:
            list1.extend(df2[f"p(E|{field})"].dropna().tolist())
            list2.extend(df2[f"p({field}|E)"].dropna().tolist())

        print(f"{model_name}: inverse v.s. posterior, Spearman")
        spearman = stats.spearmanr(list1, list2).statistic
        spearman_pvalue = fisher_pvalue_spear(list1, list2)

        print(f"{model_name}: inverse v.s. posterior, Pearson")
        pearson = stats.pearsonr(list1, list2).statistic
        pearson_pvalue = fisher_pvalue_pear(list1, list2)

        out = {"model_name": model_name,
               "spearman": spearman,
               "spearman_pvalue": spearman_pvalue,
               "pearson": pearson,
               "pearson_pvalue": pearson_pvalue}
        list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output_fp = os.path.join(output_dir, "3inverse_posterior.csv")
    out_df.to_csv(output_fp, index=False)


def main4(posterior_fp, output_dir):
    df = pd.read_csv(posterior_fp)
    list_outs = []
    for model_name in df["model_name"].unique():
        df2 = df[df["model_name"] == model_name]
        list1 = []
        list2 = []
        for field in ["A", "B", "C"]:
            list1.extend(df2[f"p(E|{field})"].dropna().tolist())
            list2.extend(df2[f"sim_{field}"].dropna().tolist())

        print(f"{model_name}: inverse v.s. similarity, Spearman")
        spearman = stats.spearmanr(list1, list2).statistic
        spearman_pvalue = fisher_pvalue_spear(list1, list2)

        print(f"{model_name}: inverse v.s. similarity, Pearson")
        pearson = stats.pearsonr(list1, list2).statistic
        pearson_pvalue = fisher_pvalue_pear(list1, list2)

        out = {"model_name": model_name,
               "spearman": spearman,
               "spearman_pvalue": spearman_pvalue,
               "pearson": pearson,
               "pearson_pvalue": pearson_pvalue}
        list_outs.append(out)

    out_df = pd.DataFrame(list_outs)
    output_fp = os.path.join(output_dir, "4inverse_similarity.csv")
    out_df.to_csv(output_fp, index=False)


def main5(posterior_fp, list_models, output_dir):
    df = pd.read_csv(posterior_fp)
    for number in [2, 3, 4, 5]:
        list_outs = []
        for model_name in list_models:
            df2 = df[df["model_name"] == model_name]
            list1 = []
            list2 = []
            for field in ["A", "B", "C"]:
                list1.extend(df2[f"p({field})"].dropna().tolist())
                list2.extend(df2[f"p({field}|E)_{number}"].dropna().tolist())

            print(f"{model_name}: prior v.s. posterior{number}, Spearman")
            spearman = stats.spearmanr(list1, list2).statistic
            spearman_pvalue = fisher_pvalue_spear(list1, list2)

            print(f"{model_name}: prior v.s. posterior{number}, Pearson")
            pearson = stats.pearsonr(list1, list2).statistic
            pearson_pvalue = fisher_pvalue_pear(list1, list2)

            out = {"model_name": model_name,
                   "spearman": spearman,
                   "spearman_pvalue": spearman_pvalue,
                   "pearson": pearson,
                   "pearson_pvalue": pearson_pvalue}
            list_outs.append(out)

        out_df = pd.DataFrame(list_outs)
        output_fp = os.path.join(output_dir, f"{4+number-1}prior_posterior{number}.csv")
        out_df.to_csv(output_fp, index=False)


def main6(posterior_fp, list_models, output_dir):
    df = pd.read_csv(posterior_fp)
    for number in [2, 3, 4, 5]:
        list_outs = []
        for model_name in list_models:
            df2 = df[df["model_name"] == model_name]
            list1 = []
            list2 = []
            for field in ["A", "B", "C"]:
                list1.extend(df2[f"sim_{field}"].dropna().tolist())
                list2.extend(df2[f"p({field}|E)_{number}"].dropna().tolist())

            print(f"{model_name}: Similarity v.s. posterior{number}, Spearman")
            spearman = stats.spearmanr(list1, list2).statistic
            spearman_pvalue = fisher_pvalue_spear(list1, list2)

            print(f"{model_name}: Similarity v.s. posterior{number}, Pearson")
            pearson = stats.pearsonr(list1, list2).statistic
            pearson_pvalue = fisher_pvalue_pear(list1, list2)

            out = {"model_name": model_name,
                   "spearman": spearman,
                   "spearman_pvalue": spearman_pvalue,
                   "pearson": pearson,
                   "pearson_pvalue": pearson_pvalue}
            list_outs.append(out)

        out_df = pd.DataFrame(list_outs)
        output_fp = os.path.join(output_dir, f"{8+number-1}similarity_posterior{number}.csv")
        out_df.to_csv(output_fp, index=False)


if __name__ == "__main__":
    exp_posterior_fp = "../1combine/1combine.csv"
    exp_output_dir = "../2correlation"
    os.makedirs(exp_output_dir, exist_ok=True)

    # main1(exp_posterior_fp, exp_output_dir)
    # main2(exp_posterior_fp, exp_output_dir)
    # main3(exp_posterior_fp, exp_output_dir)
    main4(exp_posterior_fp, exp_output_dir)

    exp_list_models = ["gpt-4o"]
    main5(exp_posterior_fp, exp_list_models, exp_output_dir)
    main6(exp_posterior_fp, exp_list_models, exp_output_dir)
