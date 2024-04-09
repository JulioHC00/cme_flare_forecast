from .base_logger import BaseLogger
from ..metrics.utils.metrics_factory import metrics_factory
from ..plotting.utils.plotter_factory import plotters_factory
import wandb
from typing import Union
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.figure
import time

matplotlib.use("Agg")


class WandBLogger(BaseLogger):
    """
    A logger class for logging metrics and loss to Weights and Biases (WandB).

    Attributes:
        config (Dict): The configuration dictionary.
        metrics (Dict): Dictionary containing initialized metrics objects.

    Methods:
        log_metrics(y_actual: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], mode: str):
            Logs metrics to WandB.
        log_loss(loss: float, mode: str = "train"):
            Logs loss to WandB.
    """

    def __init__(
        self,
        project: str,
        full_config: DictConfig,
        metrics_conf: DictConfig,
        plotter_conf: DictConfig,
        best_run_metric: str,
        device: str,
        mode: str = "online",
    ):
        """
        Initialize the WandBLogger class.

        Parameters:
            config (Dict): The configuration dictionary.
        """
        self.project = project
        self.full_config = full_config
        self.mode = mode
        self.device = device

        # Initialize WandB
        wandb.init(project=self.project, config=self.full_config, mode=self.mode)  # type: ignore

        self.metrics_conf = metrics_conf
        self.plotter_conf = plotter_conf

        self.best_run_metric = best_run_metric

        # Check if best_run_metric is in the metrics_conf
        if self.best_run_metric not in self.metrics_conf:
            raise ValueError(
                f"best_run_metric {best_run_metric} not in metrics_conf. Valid metrics are {self.metrics_conf.keys()}"
            )

        # Initialize metrics using a factory function
        self.metrics = metrics_factory(self.metrics_conf)

        # Check the best_run_metric object has the is_better method
        if not hasattr(self.metrics[self.best_run_metric], "is_better"):
            raise AttributeError(
                f"best_run_metric {best_run_metric} does not have the is_better method"
            )

        # Initialize the plotters
        self.plotters = plotters_factory(self.plotter_conf)

        # We'll keep track of the best metrics
        self.best_metrics = {}

        # Holds the current best value for the best_run_metric
        self.best_metric_so_far = None

        # Holds the summary we return at the end with the metrics at the epoch for which the best_metric_value is best
        self.best_run_summary = None

        self.is_currently_best = False

    def log_metrics(
        self,
        y_actual: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        mode: str,
    ) -> None:
        """
        Logs metrics to WandB.

        Parameters:
            y_actual (Union[np.ndarray, torch.Tensor]): Ground truth values.
            y_pred (Union[np.ndarray, torch.Tensor]): Predicted values.
            mode (str): Mode in which the metrics are computed ("train", "val", "test").
        """
        start = time.time()

        log_dict = {}

        if len(self.metrics) == 0:
            return

        for metric_name, metric_obj in self.metrics.items():
            # Update metric object with new data
            metric_obj.update(targets=y_actual, preds=y_pred)

            # Compute the metric and log it
            new_metric = metric_obj.compute()
            log_dict[f"{metric_name}_{mode}"] = new_metric

            # Update the best metrics
            # Only objects with the is_better method will be updated
            if not hasattr(metric_obj, "is_better"):
                continue
            else:
                best_metric_name = f"best_{metric_name}_{mode}"
                if best_metric_name not in self.best_metrics:
                    self.best_metrics[best_metric_name] = new_metric
                else:
                    if metric_obj.is_better(
                        new_metric, self.best_metrics[best_metric_name]
                    ):
                        self.best_metrics[best_metric_name] = new_metric

            # If this is the best_run_metric we need to do some more things
            if metric_name == self.best_run_metric:
                # Then we need to check if it's better than the best so far
                if self.best_metric_so_far is None:
                    self.best_metric_so_far = new_metric
                    self.is_currently_best = True
                elif metric_obj.is_better(new_metric, self.best_metric_so_far):
                    self.best_metric_so_far = new_metric
                    self.is_currently_best = True
                else:
                    self.is_currently_best = False

        get_metrics_time = time.time()
        took_get_metrics = get_metrics_time - start

        # If this run is currently the best, need to update the summary

        if self.is_currently_best:
            # Only add things that can be converted to a float
            self.best_run_summary = {}
            for key, value in log_dict.items():
                try:
                    fl_value = float(value)
                    self.best_run_summary[key] = fl_value
                except TypeError:
                    continue

        wandb.log(log_dict)

        log_metrics_time = time.time()

        took_log_metrics = log_metrics_time - get_metrics_time

        total_time = log_metrics_time - start

        print(f"Took {took_get_metrics} to get metrics")
        print(f"Took {took_log_metrics} to log metrics")
        print(f"Took {total_time} to log metrics")

    def log_loss(self, loss: float, mode: str) -> None:
        """
        Logs loss to WandB.

        Parameters:
            loss (float): The loss value to log.
            mode (str): Mode in which the loss is computed ("train", "val", "test").
        """
        wandb.log({f"loss_{mode}": loss})

    def log_plots(self, metadata: pd.DataFrame, mode: str) -> None:
        start = time.time()
        fig_dict = {}

        if len(self.plotters) == 0:
            return

        for plotter_name, plotter_obj in self.plotters.items():
            fig = plotter_obj.plot(metadata, mode)

            if isinstance(fig, dict):
                for key, value in fig.items():
                    fig_dict[f"{plotter_name}_{mode}_{key}"] = wandb.Image(value)

                    value.clf()
                    plt.close(value)

            else:
                fig_dict[plotter_name] = wandb.Image(fig)
                fig.clf()
                plt.close(fig)

        get_plots_time = time.time()

        took_get_plots = get_plots_time - start

        # Log the figures with the epoch
        wandb.log(fig_dict)

        log_plots_time = time.time()

        took_log_plots = log_plots_time - get_plots_time

        total_time = log_plots_time - start

        print(f"Took {took_get_plots} to get plots")
        print(f"Took {took_log_plots} to log plots")
        print(f"Took {total_time} to log plots")

    def reset_metrics(self) -> None:
        """
        Resets the metrics.
        """
        for metric_obj in self.metrics.values():
            metric_obj.reset()

        for plotter_obj in self.plotters.values():
            plotter_obj.reset()

    def get_summary(self) -> dict:
        """
        Returns the summary of the best run.
        """
        if self.best_run_summary is None:
            raise ValueError("No best run summary found")

        return self.best_run_summary

    def log_model_params(self, model) -> None:
        """
        Logs the model parameters to WandB.
        Parameters:
            model (torch.nn.Module): The model to log.
        """
        start = time.time()
        wandb.watch(model, log="all")
        time_took = time.time() - start

        print(f"Took {time_took} to log model params")

    def log_learning_rate(self, learning_rate) -> None:
        wandb.log({"lr": learning_rate})

    def finish(self) -> None:
        # Finally we log the best metrics
        for best_metric_name, best_metric_value in self.best_metrics.items():
            try:
                wandb.run.summary[best_metric_name] = best_metric_value
            except TypeError:
                continue

        wandb.finish()
