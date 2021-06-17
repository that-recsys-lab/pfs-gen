import argparse
import json
import pickle
import matplotlib.pyplot as plt

from clearml import Task
from scipy.stats import ks_2samp, ttest_ind

from datasets import Dataset, get_dataset
from generators import SyntheticDataGenerator


def log_it(t: Task, title, key, value, key_prefix):
    if isinstance(value, int) or isinstance(value, float):
        t.logger.report_scalar(key_prefix+key, title, value, 0)
    elif isinstance(value, plt.Figure):
        t.logger.report_matplotlib_figure(key_prefix+key, title, figure=value,
                                          iteration=0)


def log_dataset_metrics(task: Task, ds: Dataset, title=None, key_prefix=""):
    stats = ds.print_stats()
    for k, v in stats:
        log_it(task, title, k, v, key_prefix)
    task.flush()


def log_difference(task: Task, ds, ds_original):
    score = ks_2samp(list(ds.rating_matrix.data), list(ds_original.rating_matrix.data)).statistic
    ttest = ttest_ind(list(ds.rating_matrix.data), list(ds_original.rating_matrix.data))

    print("ks score: ", score)
    print("t-test p-value: ", ttest.pvalue)

    task.logger.report_scalar("ks-score", "", score, 0)
    task.logger.report_scalar("rating mean t-test", "", ttest.pvalue, 0)
    task.logger.report_scalar("rating mean t-test stat", "", ttest.statistic, 0)


def main(task, generator_task_id, extra):
    gen_task = Task.get_task(task_id=generator_task_id)
    generator_path = gen_task.artifacts["generator"].get_local_copy()
    original_ds_name = gen_task._hyper_params_manager.get_hyper_params(
        sections=['Args'], projector=None
    )['Args']['dataset']['value']
    original_ds = get_dataset(original_ds_name)

    with open(generator_path, "rb") as f:
        generator_obj = pickle.load(f)  # type: SyntheticDataGenerator
    generated = generator_obj.generate(task, **extra)

    with open('dataset.csv', "w") as f:
        for u, i, r in generated.as_iterator():
            f.write("%d,%d,%d\n" % (u, i, int(r * 2.5 + 2.5)))

    task.upload_artifact('dataset', artifact_object='dataset.csv', delete_after_upload=True)

    log_dataset_metrics(task, generated, "synthetic", "synthetic ")
    log_dataset_metrics(task, original_ds, "original", "original ")
    log_difference(task, generated, original_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_id', type=str, action='store', required=True)
    parser.add_argument('--parameters', type=str, action='store', required=False, default="{}")
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()
    extra_args = json.loads(args.parameters)

    tags = ["GenerateData"]
    if args.final:
        tags.append("Final")

    task = Task.init(project_name="Synthetic data generators", task_name="generate_data", tags=tags,
                     reuse_last_task_id=not args.final, task_type=Task.TaskTypes.inference)

    main(task, args.task_id, extra_args)
