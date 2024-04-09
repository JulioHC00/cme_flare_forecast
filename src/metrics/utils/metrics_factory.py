from .. import metrics


def metrics_factory(metrics_config) -> dict:
    all_metrics = {}
    for metrics_name in metrics_config.keys():
        if metrics_name == "None":
            return all_metrics
        else:
            try:
                metric_class = getattr(metrics, metrics_name)
            except AttributeError:
                raise AttributeError(f"Metric {metrics_name} not implemented")

            metric_params = metrics_config[metrics_name]["args"]
            metric = metric_class(**metric_params)

            all_metrics[metrics_name] = metric

    return all_metrics
