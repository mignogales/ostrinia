import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets.ostrinia import Ostrinia


def as_dataframe(values, index=None):
    if isinstance(values, pd.DataFrame):
        return values
    if isinstance(values, pd.Series):
        return values.to_frame()

    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]

    return pd.DataFrame(array, index=index)


def load_temperature(dataset, temperature_key):
    raw_extra_data = getattr(dataset, "raw_extra_data", {})

    if temperature_key in raw_extra_data:
        return raw_extra_data[temperature_key]

    if temperature_key in dataset.extra_data:
        return dataset.extra_data[temperature_key]

    if dataset.target == temperature_key:
        return dataset.dataframe()

    available = sorted(dataset.extra_data.keys())
    raise KeyError(
        f"Could not find '{temperature_key}'. Available extra_data keys: {available}. "
        f"Dataset target is '{dataset.target}'."
    )


def plot_temperature(temperature, output_path, node=None):
    temperature = temperature.copy()
    temperature.index = pd.to_datetime(temperature.index)

    fig, ax = plt.subplots(figsize=(14, 5))

    if node is None:
        temperature.plot(ax=ax, color="0.75", alpha=0.35, legend=False, linewidth=0.8)
        temperature.mean(axis=1).plot(
            ax=ax,
            color="tab:red",
            linewidth=2.0,
            label="Mean temperature",
        )
    else:
        temperature.iloc[:, node].plot(
            ax=ax,
            color="tab:red",
            linewidth=1.6,
            label=f"Node {node}",
        )

    ax.set_title("Ostrinia temperature")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot temperature from the Ostrinia dataset.")
    parser.add_argument("--root", default="datasets", help="Dataset root folder.")
    parser.add_argument("--target", default="incrementing_ostrinia", help="Ostrinia target used to initialise the dataset.")
    parser.add_argument("--temperature-key", default="TempAv", help="Temperature key in dataset.extra_data.")
    parser.add_argument("--node", type=int, default=None, help="Optional node index to plot instead of all nodes.")
    parser.add_argument("--output", default="ostrinia_temperature.png", help="Output image path.")
    args = parser.parse_args()

    dataset = Ostrinia(
        root=args.root,
        target=args.target,
        add_second_target=False,
        spatial_information=True,
    )

    temperature = load_temperature(dataset, args.temperature_key)
    temperature = as_dataframe(temperature, index=dataset.dataframe().index)

    plot_temperature(
        temperature=temperature,
        output_path=Path(args.output),
        node=args.node,
    )

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
