# -*- encoding: utf-8 -*-
import os
import argparse
import pandas as pd


def filter_data(path):
    df = pd.read_csv(path, sep="\t", header=None)

    filter_func = lambda x: len(x.replace('<EOS>', '').replace('<EOP>', '').strip()) > 0
    new_df = df.loc[df[1].apply(filter_func) & df[2].apply(filter_func)]
    new_df.to_csv(path, index=False, header=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="data directory")

    args = parser.parse_args()
    filter_data(os.path.join(args.data_dir, "train.tsv"))
    filter_data(os.path.join(args.data_dir, "dev.tsv"))
    filter_data(os.path.join(args.data_dir, "test.tsv"))
