from clearml import Task
from datasets import Dataset


class SyntheticDataGenerator:
    def build(self, task: Task, base_dataset: Dataset, **kwargs):
        raise NotImplementedError()

    def generate(self, task: Task, n_users=None, **kwargs) -> Dataset:
        raise NotImplementedError()
