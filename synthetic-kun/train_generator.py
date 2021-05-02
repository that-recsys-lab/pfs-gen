import argparse
import json
import pickle

from clearml import Task
import datasets
import generators
from generators import SyntheticDataGenerator


def main(task, dataset_name, generator_name, extra):
    if dataset_name not in datasets.name_to_class:
        raise Exception("Unknown dataset name")
    if generator_name not in generators.name_to_class:
        raise Exception("Unknown generator name")
    dataset_obj = datasets.name_to_class[dataset_name]()
    generator_obj = generators.name_to_class[generator_name](dataset_obj.rating_matrix)  # type: SyntheticDataGenerator
    generator_obj.build(task, **extra)
    with open("generator.pickle", "wb") as f:
        pickle.dump(generator_obj, f)
    task.upload_artifact('generator', artifact_object='generator.pickle', delete_after_upload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, action='store', required=True)
    parser.add_argument('--generator', type=str, action='store', required=True)
    parser.add_argument('--parameters', type=str, action='store', required=False, default="{}")
    parser.add_argument('--final', action='store_true')

    args = parser.parse_args()
    extra_args = json.loads(args.parameters)

    tags = ["TrainGenerator"]
    if args.final:
        tags.append("Final")

    task = Task.init(project_name="Synthetic data generators", task_name="train_generator", tags=tags,
                     reuse_last_task_id=False)

    main(task, args.dataset, args.generator, extra_args)

