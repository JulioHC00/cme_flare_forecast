from .. import plotters


def plotters_factory(plotters_config) -> dict:
    all_plotters = {}
    for plotter_name in plotters_config.keys():
        if plotter_name == "None":
            return all_plotters
        else:
            try:
                plotter_class = getattr(plotters, plotter_name)
            except AttributeError:
                raise AttributeError(f"Plotter {plotter_name} not implemented")

            plotter_params = plotters_config[plotter_name]["args"]
            plotter = plotter_class(**plotter_params)

            all_plotters[plotter_name] = plotter

    return all_plotters
