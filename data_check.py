# -*- encoding: utf-8 -*-
import os
import argparse
import pandas as pd


def sanity_check(path):

    h_idx, b_idx = [], []
    df = pd.read_csv(path, sep="\t")

    for idx, row in df.iterrows():
        headline = row[1]
        bodytext = row[2]
        h = headline.replace('<EOS>', '').replace('<EOP>', '').strip()
        b = bodytext.replace('<EOP>', '').replace('<EOS>', '').strip()
        if len(h) == 0:
            h_idx.append(idx)
        if len(b) == 0:
            b_idx.append(idx)

    return h_idx, b_idx, len(h_idx), len(b_idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="data directory")

    args = parser.parse_args()
    print(sanity_check(os.path.join(args.data_dir, "train.tsv")))
    print(sanity_check(os.path.join(args.data_dir, "dev.tsv")))
    print(sanity_check(os.path.join(args.data_dir, "test.tsv")))
