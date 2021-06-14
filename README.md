# pfs-gen
Data generation by probabilistic factor sampling

## Architecture

We divided the experiments into three stages (presentation mentions two, but we decided to rework that for better design). Each step has its own script.

\begin{itemize}
    \item Generator training. This step inputs a dataset name, generator name, and a set of arbitrary generator-specific parameters.
    \item Dataset generation. This step is required, as we may want to generate several datasets from a single trained generator. This step inputs task (experiment) of generator training experiment, and a set of generator-specific parameters. Metrics are computed and logged as well.
    \item Recommender system evaluation. This step inputs dataset task id, or name (for non-synthetic datasets); recommender system name; and a set of recommender system-specific parameters. This step uses LibrecAuto.
\end{itemize}

That design allows easy extensibility and use. Previously, we had first two experiement types clumped together, and training multiple experiments in one experiment for the latter one. It was hard to extend and use - experimental system was barely used, as it was getting in the way of doing research. Moreover, having "Recommender system evaluation" experiment type can be reused by other labmates, who also research recommender systems.

Other than these scripts, there are classes and files that represent logic for dataset generation, and dataset loading.

Class SyntheticDataGenerator represents a generic interface for a dataset generator. A more specific class MFDataGenerator represents generators based on matrix factorization. My random walk and FSPIRE generators implement these interfaces.

dataset.py contains dataset loading logic. Each dataset type is supposed to implement this interface, including synthetic dataset.

To interface with LibrecAuto, we write a configuration file in required format and call Python command. It trains recommender system and outputs metrics into a csv file, which is then read.

## Set-up

Install required libraries (pandas, numpy, clearml, torch).

Put Movielens file (ml-1m-ratings.dat) into datasets/ folder.

Set up ClearML account, and set up credentials:
https://allegro.ai/clearml/docs/docs/getting_started/getting_started_clearml_hosted_service.html

## Re-running experiments

Training generators:

```
python3 train_generator.py --dataset ml1m --generator fspire_mixed --parameters='{"epochs":50}' --final
python3 train_generator.py --dataset ml1m --generator random_walk --final
```

Synthesizing datasets:

```
python3 generate_data.py --task_id <FSPIRE experiment id> --final
python3 generate_data.py --task_id <RandomWalk experiment id> --final
```

Training recommender systems and getting metrics:

```

python3 evaluate_recommender.py --task_id <FSPIRE dataset id> --recommender svdpp --parameters '{"num-factors":200,"iterator-max":200}' --final
python3 evaluate_recommender.py --task_id <FSPIRE dataset id> --recommender bpr --parameters '{"num-topics":200,"iterator-max":200}' --final
python3 evaluate_recommender.py --task_id <FSPIRE dataset id> --recommender itemknn --parameters '{"neighborhood-size":25,"shrinkage":10}' --final
python3 evaluate_recommender.py --task_id <FSPIRE dataset id> --recommender biasedmf --parameters '{"num-factors":200,"iterator-max":200,"item-reg":0.01}' --final

python3 evaluate_recommender.py --task_id <RandomWalk dataset id> --recommender svdpp --parameters '{"num-factors":200,"iterator-max":200}' --final
python3 evaluate_recommender.py --task_id <RandomWalk dataset id> --recommender bpr --parameters '{"num-topics":200,"iterator-max":200}' --final
python3 evaluate_recommender.py --task_id <RandomWalk dataset id> --recommender itemknn --parameters '{"neighborhood-size":25,"shrinkage":10}' --final
python3 evaluate_recommender.py --task_id <RandomWalk dataset id> --recommender biasedmf --parameters '{"num-factors":200,"iterator-max":200,"item-reg":0.01}' --final

```

This will add 8 experiments. Numbers for the results tables can be viewed in the ClearML dashboard.
