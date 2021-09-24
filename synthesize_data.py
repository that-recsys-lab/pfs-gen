import argparse

from scipy.stats import ks_2samp
from datasets import load_movielens_ratings, load_amazon_reviews
from generator.torch_model import TorchDataGenerator
import os.path
import pickle
import numpy as np

configs = [
    (x, True, [0.1] + list(np.arange(1.0, 30.0, 3.0))) for x in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
] + [
    (x, False, [0.1] + list(np.arange(1.0, 30.0, 3.0))) for x in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
]


def crossvalidate_data(dataset_name):
    data_filename = f"data_{dataset_name}.pickle"
    mf_filename = f"mf_{dataset_name}.pickle"
    model_filename = f"trained_model_{dataset_name}.pickle"

    if not os.path.exists(data_filename):
        with open(data_filename, "rb") as f:
            rating_matrix = pickle.load(f)
            id_to_user = pickle.load(f)
            id_to_item = pickle.load(f)
    else:
        if dataset_name == "ml":
            ratings = load_movielens_ratings()
        elif dataset_name == "amazon":
            ratings = load_amazon_reviews()
        else:
            raise Exception("Dataset name should be <ml> or <amazon>")
        rating_matrix, id_to_user, id_to_item = ratings

        with open(data_filename, "wb") as f:
            pickle.dump(rating_matrix, f)
            pickle.dump(id_to_user, f)
            pickle.dump(id_to_item, f)

    if not os.path.exists(mf_filename):
        with open(mf_filename, "rb") as f:
            mf = pickle.load(f)
    else:
        print("Rebuilding MF")
        mf = TorchDataGenerator(rating_matrix, 20)
        mf._build_mf(200)

        with open(mf_filename, "wb") as f:
            pickle.dump(mf, f)

    best_model = None
    best_score = 1.0

    for logloss_coef, ilc, scales in configs:
        with open(mf_filename, "rb") as f:
            mf = pickle.load(f)

        mf.verbose = True
        mf.epochs = 50
        mf._build_torch_model(200)
        mf._train_torch_model(logloss_coef, lr=4e-3, implicit_logloss=ilc)

        mf.user_vectors = mf.model.user_factors.weight.detach().numpy()
        mf.item_vectors = mf.model.item_factors.weight.detach().numpy()
        mf.item_proba_vectors = mf.model.item_implicit_factors.weight.detach().numpy()
        mf._build_gmm(10)

        for coef in scales:
            generated = mf.generate(300, False, False, implicit_coef=coef)
            score = ks_2samp(list(generated.values()), list(rating_matrix.data)).statistic
            print(score, logloss_coef, ilc, coef)
            if score < best_score:
                best_model = mf
                best_score = score

    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)
    return best_model


def generate(dataset_name):
    data_filename = f"experiments/data/generated_{dataset_name}.csv"
    model_filename = f"trained_model_{dataset_name}.pickle"

    with open(model_filename, "rb") as f:
        mf = pickle.load(f)

    generated = mf.generate(mf.n_users, False, False, implicit_coef=mf._implicit_coef)

    with open(data_filename, "w") as f:
        for k, v in generated.items():
            f.write("%d,%d,%d\n" % (k[0], k[1], int(v * 2.5 + 2.5)))


def plot(dataset_name):
    model_filename = f"trained_model_{dataset_name}.pickle"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
                        type=str, action='store', required=True)
    parser.add_argument('-t', '--train',
                        type=bool, action='store', default=True)
    parser.add_argument('-g', '--generate',
                        type=bool, action='store', default=True)
    parser.add_argument('-p', '--plot',
                        type=bool, action='store', default=True)

    args = parser.parse_args()
    if args.user is None:
        args.user = args.directory

    if args.train:
        crossvalidate_data(args.dataset)

    if args.generate:
        generate(args.dataset)

    if args.plot:
        plot(args.dataset)
