import pandas as pd
import torch


def convert_df_tensors_to_cpu(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if isinstance(df[col][0], torch.Tensor):
            df[col] = df[col].apply(lambda x: x.cpu())
    return df


def convert_dict_tensors_to_cpu(d: dict) -> dict:
    for key in d.keys():
        if isinstance(d[key], torch.Tensor):
            d[key] = d[key].cpu()
    return d
