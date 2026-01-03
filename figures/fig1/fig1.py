import io
import cairosvg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def svg_to_png_array(svg_string, scale=1.0, color=None):
    """
    Convert SVG string → PNG bytes → NumPy array for plotting.
    Optionally replace the stroke color in the SVG string.
    """
    if color:
        # Replace the stroke color in the SVG string
        svg_string = svg_string.replace("stroke='black'", f"stroke='{color}'")
    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode("utf-8"), scale=scale)
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    return np.array(img)


def render_png_panel(ax, img_array, title=None, fontdict=None):
    """
    Render a PNG array into a Matplotlib axis.
    """
    ax.imshow(img_array)
    ax.axis("off")
    if title:
        ax.set_title(title, fontdict=fontdict)


def plot_reconstruction_grid_cairo(input_svgs, reconstructions_dict, figsize=(12, 4), scale=1.0):
    """
    input_svgs: list of SVG strings (ground truth)
    reconstructions_dict: { model_name: [svg_string, ...] }
    """
    num_inputs = len(input_svgs)
    model_names = list(reconstructions_dict.keys())
    num_models = len(model_names)

    # Define colors for each model
    model_colors = {
        "SVG AE": "orange",
        "MAE": "blue",
        "JEPA": "green",
        "Contrastive": "purple",
    }

    # Define font properties for model names
    fontdict = {
        "fontsize": 10,
        "fontweight": "normal",
        "family": "Times New Roman",
        "color": "black",
    }

    fig, axes = plt.subplots(num_inputs, num_models + 1, figsize=figsize, squeeze=False)

    for i, inp_svg in enumerate(input_svgs):
        # Convert input SVG → PNG array
        inp_png = svg_to_png_array(inp_svg, scale=scale)
        render_png_panel(axes[i, 0], inp_png)  # No title for input column

        for j, model in enumerate(model_names):
            recon_svg = reconstructions_dict[model][i]
            recon_png = svg_to_png_array(
                recon_svg, scale=scale, color=model_colors.get(model, "black")
            )
            title = model if i == 0 else None
            render_png_panel(axes[i, j + 1], recon_png, title, fontdict=fontdict)

    fig.tight_layout()
    return fig


def main():
    input_svgs = [
        "<svg width='64' height='64' viewBox='0 0 64 64'><circle cx='32' cy='32' r='20' stroke='black' fill='none'/></svg>",
        "<svg width='64' height='64' viewBox='0 0 64 64'><rect x='10' y='10' width='40' height='40' stroke='black' fill='none'/></svg>",
    ]

    reconstructions = {
        "SVG AE": [
            "<svg width='64' height='64' viewBox='0 0 64 64'><circle cx='30' cy='30' r='22' stroke='black' fill='none'/></svg>",
            "<svg width='64' height='64' viewBox='0 0 64 64'><rect x='12' y='12' width='38' height='38' stroke='black' fill='none'/></svg>",
        ],
        "MAE": [
            "<svg width='64' height='64' viewBox='0 0 64 64'><circle cx='32' cy='32' r='18' stroke='black' fill='none'/></svg>",
            "<svg width='64' height='64' viewBox='0 0 64 64'><rect x='11' y='14' width='40' height='36' stroke='black' fill='none'/></svg>",
        ],
        "JEPA": [
            "<svg width='64' height='64' viewBox='0 0 64 64'><circle cx='32' cy='32' r='20' stroke='black' fill='none'/></svg>",
            "<svg width='64' height='64' viewBox='0 0 64 64'><rect x='10' y='10' width='42' height='42' stroke='black' fill='none'/></svg>",
        ],
        "Contrastive": [
            "<svg width='64' height='64' viewBox='0 0 64 64'><circle cx='31' cy='31' r='19' stroke='black' fill='none'/></svg>",
            "<svg width='64' height='64' viewBox='0 0 64 64'><rect x='13' y='13' width='36' height='36' stroke='black' fill='none'/></svg>",
        ],
    }

    plot_reconstruction_grid_cairo(input_svgs, reconstructions, figsize=(10, 6), scale=2.0)

    # Save the figure as fig1.png
    plt.savefig("fig1.png", dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
