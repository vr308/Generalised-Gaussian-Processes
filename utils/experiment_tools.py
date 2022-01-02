#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper methods to streamline experiments and persistence of results/logs

"""
import utils.dataset as uci_datasets
from utils.dataset import Dataset
from typing import Type

class ExperimentName:
    def __init__(self, base):
        self.s = base

    def add(self, name, value):
        self.s += f"_{name}-{value}"
        return self

    def get(self):
        return self.s

def experiment_name(
    date_str,
    dataset_name,
    model_name,
    split_index,
    train_test_split,
    num_inducing,
    max_iter,
    num_epochs,
    batch_size
):
    if model_name == 'SGPR':
        return (
            ExperimentName(date_str)
            .add("dataset", dataset_name)
            .add("model_name", model_name)
            .add("split", split_index)
            .add("frac", train_test_split)
            .add("num_inducing", num_inducing)
            .add("max_iter", max_iter)
            .get())
    elif model_name == 'SVGP':
         return (
           ExperimentName(date_str)
           .add("dataset", dataset_name)
           .add("model_name", model_name)
           .add("split", split_index)
           .add("frac", train_test_split)
           .add("num_inducing", num_inducing)
           .add("num_epochs", num_epochs)
           .add("batch_size", batch_size)
           .get())

def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)
