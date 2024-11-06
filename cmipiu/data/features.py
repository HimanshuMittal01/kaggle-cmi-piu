"""
Module responsible for feature engineering
"""

from typing import Any

import numpy as np
import polars as pl

from sklearn.preprocessing import OrdinalEncoder


def __add_fgc_bia_tvalues(
    df: pl.DataFrame, training: bool = False, artifacts: dict[str, Any] = {}
) -> pl.DataFrame:
    """
    Creates tvalue for FGC and BIA numerical columns.

    Artifacts Create/Use:
    - meanstd_values: pl.DataFrame
    """
    fgc_mags = [
        col
        for col in df.columns
        if col.startswith("FGC") and col != "FGC-Season" and not col.endswith("Zone")
    ]
    bia_mags = [
        col
        for col in df.columns
        if col.startswith("BIA")
        and col != "BIA-Season"
        and col != "BIA-BIA_Activity_Level_num"
        and col != "BIA-BIA_Frame_num"
    ]
    cols_to_transform = fgc_mags + bia_mags

    def make_new_wh(df: pl.DataFrame):
        return df.with_columns(
            neww=pl.col("Physical-Weight") // 10, newh=pl.col("Physical-Height") // 6
        )

    def make_meanstd_cols(
        df: pl.DataFrame,
        training: bool = False,
        meanstd_values: pl.DataFrame | None = None,
    ):
        if training:
            return df.with_columns(
                [
                    pl.col(col)
                    .mean()
                    .over(["Basic_Demos-Age", "Basic_Demos-Sex", "neww", "newh"])
                    .alias(f"m{col}")
                    for col in cols_to_transform
                ]
                + [
                    pl.col(col)
                    .std()
                    .over(["Basic_Demos-Age", "Basic_Demos-Sex", "neww", "newh"])
                    .fill_null(1)
                    .alias(f"s{col}")
                    for col in cols_to_transform
                ],
            )

        else:
            assert (
                meanstd_values is not None
            ), "Artifacts must contain 'meanstd_values' when not training."
            return df.join(
                meanstd_values,
                on=["Basic_Demos-Age", "Basic_Demos-Sex", "neww", "newh"],
                how="left",
            )

    def make_tvalues(df: pl.DataFrame):
        return df.with_columns(
            [
                ((pl.col(col) - pl.col(f"m{col}")) / (pl.col(f"s{col}") + 1e-7)).alias(
                    f"t{col}"
                )
                for col in cols_to_transform
            ]
        )

    df = (
        df.pipe(make_new_wh)
        .pipe(
            make_meanstd_cols,
            training=training,
            meanstd_values=artifacts.get("meanstd_values"),
        )
        .pipe(make_tvalues)
    )

    # Save meanstd_values if training
    if training:
        artifacts["meanstd_values"] = df.select(
            pl.col(
                ["Basic_Demos-Age", "Basic_Demos-Sex", "neww", "newh"]
                + [f"m{col}" for col in cols_to_transform]
                + [f"s{col}" for col in cols_to_transform]
            )
        ).unique()

    # Drop temporary created mean and std values
    df = df.drop(
        ["neww", "newh"]
        + [f"m{col}" for col in cols_to_transform]
        + [f"s{col}" for col in cols_to_transform]
    )

    return df


def __create_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add interaction features and missing indicator for Physical-Waist_Circumference
    """
    # Add missing indicator for waist circumference
    df = df.with_columns(
        (
            pl.col("Physical-Waist_Circumference")
            .is_null()
            .alias("missingindicator_Waist_Circumference")
        ),
    )

    # Create interaction features
    df = df.with_columns(
        (pl.col("Physical-BMI") * pl.col("Basic_Demos-Age")).alias("BMI_Age"),
        (
            pl.col("PreInt_EduHx-computerinternet_hoursday") * pl.col("Basic_Demos-Age")
        ).alias("Internet_Hours_Age"),
        (
            pl.col("Physical-BMI") * pl.col("PreInt_EduHx-computerinternet_hoursday")
        ).alias("BMI_Internet_Hours"),
        (pl.col("BIA-BIA_Fat") / pl.col("BIA-BIA_BMI")).alias("BFP_BMI"),
        (pl.col("BIA-BIA_FFMI") / pl.col("BIA-BIA_Fat")).alias("FFMI_BFP"),
        (pl.col("BIA-BIA_FMI") / pl.col("BIA-BIA_Fat")).alias("FMI_BFP"),
        (pl.col("BIA-BIA_LST") / pl.col("BIA-BIA_TBW")).alias("LST_TBW"),
        (pl.col("BIA-BIA_Fat") * pl.col("BIA-BIA_BMR")).alias("BFP_BMR"),
        (pl.col("BIA-BIA_Fat") * pl.col("BIA-BIA_DEE")).alias("BFP_DEE"),
        (pl.col("BIA-BIA_BMR") / pl.col("Physical-Weight")).alias("BMR_Weight"),
        (pl.col("BIA-BIA_DEE") / pl.col("Physical-Weight")).alias("DEE_Weight"),
        (pl.col("BIA-BIA_SMM") / pl.col("Physical-Height")).alias("SMM_Height"),
        (pl.col("BIA-BIA_SMM") / pl.col("BIA-BIA_FMI")).alias("Muscle_to_Fat"),
        (pl.col("BIA-BIA_TBW") / pl.col("Physical-Weight")).alias("Hydration_Status"),
        (pl.col("BIA-BIA_ICW") / pl.col("BIA-BIA_TBW")).alias("ICW_TBW"),
    )

    return df


def __transform_features(
    df: pl.DataFrame, training: bool = False, artifacts: dict[str, Any] = {}
) -> pl.DataFrame:
    """
    Transform some features like CGAS-CGAS_Score, PAQ_C-PAQ_C_Total, etc

    Artifacts Create/Use:
    - encoder: OrdinalEncoder
    """
    # Categorical bin CGAS Score
    df = df.with_columns(
        (pl.col("CGAS-CGAS_Score") // 5).alias("CGAS-CGAS_Score"),
    )

    # Make PAQ Total column and remove PAQ_C-PAQ_C_Total and PAQ_A-PAQ_A_Total
    df = df.with_columns(
        PAQ_Total=pl.when(
            (pl.col("PAQ_C-PAQ_C_Total").is_not_null())
            | (pl.col("PAQ_A-PAQ_A_Total").is_not_null())
        )
        .then(
            (
                pl.col("PAQ_C-PAQ_C_Total").fill_null(0)
                + pl.col("PAQ_A-PAQ_A_Total").fill_null(0)
            )
            / 2
        )
        .otherwise(pl.lit(None)),
    )

    # Add PAQ_A-Season to PAQ_C-Season and remove PAQ_A-Season later
    df = df.with_columns(
        pl.when(pl.col("PAQ_C-Season").is_null())
        .then(pl.col("PAQ_A-Season"))
        .otherwise(pl.col("PAQ_C-Season"))
    )

    # Encode season columns
    season_cols = [
        "Basic_Demos-Enroll_Season",
        "CGAS-Season",
        "Physical-Season",
        "Fitness_Endurance-Season",
        "FGC-Season",
        "BIA-Season",
        "PAQ_C-Season",
        "SDS-Season",
        "PreInt_EduHx-Season",
    ]
    df = df.with_columns(pl.col(season_cols).fill_null("Missing"))

    if training:
        # Fit ordinal encoder
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df[season_cols])

        # put that in encoder
        artifacts["encoder"] = encoder

    assert "encoder" in artifacts, "Artifacts must contain 'encoder' when not training."
    res = artifacts["encoder"].transform(df[season_cols])
    encoded_season_df = pl.DataFrame(
        res, schema=list(artifacts["encoder"].get_feature_names_out()), orient="row"
    )
    df = df.with_columns(encoded_season_df)

    return df


def __impute_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Impute Physical BMI * 1.2 + some noise (generated from normal distribution)

    Artifacts Create/Use:
    - encoder: OrdinalEncoder
    """
    # Impute Waist Circumference
    df = df.with_columns(
        pl.when(pl.col("Physical-Waist_Circumference").is_null())
        .then(pl.col("Physical-BMI") * 1.2 + np.random.randn())
        .otherwise(pl.col("Physical-Waist_Circumference"))
        .alias("Physical-Waist_Circumference")
    )

    # # Impute values
    # imputing_cols = [
    #     'Basic_Demos-Age', 'Basic_Demos-Sex', 'CGAS-CGAS_Score',
    #     'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',
    #     'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
    #     'Fitness_Endurance-Max_Stage', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',
    #     'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone',
    #     'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL',
    #     'FGC-FGC_TL_Zone', 'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',
    #     'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI',
    #     'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 'BIA-BIA_ICW', 'BIA-BIA_LDM',
    #     'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T',
    #     'PreInt_EduHx-computerinternet_hoursday'
    # ]
    # if training:
    #     imputer = KNNImputer(n_neighbors=10)
    #     res = imputer.fit_transform(df[imputing_cols])
    # else:
    #     assert imputer is not None
    #     res = imputer.transform(df[imputing_cols])

    # df = df.drop(imputing_cols)
    # imputed_df = pl.DataFrame(res, schema=list(imputer.get_feature_names_out()), orient="row")
    # df = pl.concat([df, imputed_df], how="horizontal")

    # if is_training:
    #     return df, imputer, encoder
    # else:
    #     return df, None, None

    return df


def feature_engineering(
    df: pl.DataFrame, training: bool = False, artifacts: dict[str, Any] = {}
):
    """
    Assembles all feature engineering steps in one
    """
    df = __add_fgc_bia_tvalues(df, training=training, artifacts=artifacts)

    # Drop rows where target is unknown
    if training:
        df = df.drop_nulls(subset=["sii"])

    df = __create_features(df)
    df = __transform_features(df, training=training, artifacts=artifacts)
    df = __impute_features(df)

    return df, artifacts


def get_features(df: pl.DataFrame) -> list[str]:
    """
    Exclude some features and return the feature columns
    """
    must_exclude_features = [
        "PAQ_A-Season",
        "PAQ_C-PAQ_C_Total",
        "PAQ_A-PAQ_A_Total",
        "PCIAT-Season",
        "PCIAT-PCIAT_01",
        "PCIAT-PCIAT_02",
        "PCIAT-PCIAT_03",
        "PCIAT-PCIAT_04",
        "PCIAT-PCIAT_05",
        "PCIAT-PCIAT_06",
        "PCIAT-PCIAT_07",
        "PCIAT-PCIAT_08",
        "PCIAT-PCIAT_09",
        "PCIAT-PCIAT_10",
        "PCIAT-PCIAT_11",
        "PCIAT-PCIAT_12",
        "PCIAT-PCIAT_13",
        "PCIAT-PCIAT_14",
        "PCIAT-PCIAT_15",
        "PCIAT-PCIAT_16",
        "PCIAT-PCIAT_17",
        "PCIAT-PCIAT_18",
        "PCIAT-PCIAT_19",
        "PCIAT-PCIAT_20",
        "PCIAT-PCIAT_Total",
        "sii",
    ]
    goodto_exclude_features = ["Fitness_Endurance-Time_Sec"]
    # letstry_exclude_features = [
    #     "Physical-BMI",
    #     "BIA-BIA_FFM",
    #     "tBIA-BIA_FFM",
    #     "tBIA-BIA_TBW",
    #     "FGC-FGC_TL_Zone",
    #     "FGC-FGC_SRR_Zone",
    #     "BIA-Season",
    #     "Basic_Demos-Enroll_Season",
    #     "CGAS-Season",
    #     "SDS-Season",
    #     "FGC-Season",
    #     "daily_avg_enmo_std",
    #     "light_0.25",
    #     "enmo_0.5",
    #     "rolling_std_anglez_0.25",
    #     "enmo_0.75",
    #     "non-wear_flag_mean",
    #     "daily_avg_enmo_max",
    #     "light_mean",
    #     "light_0.75",
    #     "daily_avg_light_mean",
    #     "daily_avg_light_max",
    #     "enmo_std",
    #     "Y_0.5",
    #     "enmo_mean",
    #     "anglez_0.5",
    #     "daily_avg_light_std",
    #     "X_0.25",
    #     "anglez_0.75",
    #     "rolling_std_anglez_std",
    #     "anglez_0.25",
    #     "X_0.5",
    #     "rolling_std_anglez_0.5",
    #     "light_0.5",
    #     "daily_avg_light_min",
    #     "rolling_std_anglez_0.75",
    # ]

    exclude_features = must_exclude_features + goodto_exclude_features

    features = df.select(pl.exclude(exclude_features)).drop("id").columns
    return features
