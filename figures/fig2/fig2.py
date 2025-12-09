from PIL import Image
import matplotlib.pyplot as plt
import io
import cairosvg


# Convert SVG string to PIL.Image
def svg_to_image(svg_code):
    png_bytes = cairosvg.svg2png(bytestring=svg_code.encode("utf-8"))
    return Image.open(io.BytesIO(png_bytes))


def make_grid(models):
    num_models = len(models)
    num_rows_per_model = 2
    num_stages = 8
    total_rows = num_models * num_rows_per_model

    fig, axes = plt.subplots(
        total_rows, num_stages, figsize=(num_stages, total_rows), gridspec_kw={"hspace": 0.1}
    )

    if total_rows == 1:
        axes = axes[None, :]
    elif num_stages == 1:
        axes = axes[:, None]

    extra_space = 0.05

    fontdict = {
        "fontsize": 10,
        "fontweight": "normal",
        "family": "Times New Roman",
        "color": "black",
    }

    for model_idx, (model_name, input_lists) in enumerate(models.items()):
        row_start = model_idx * num_rows_per_model

        # Shift model rows down by extra_space
        if model_idx > 0:
            for row_idx in range(row_start, row_start + num_rows_per_model):
                for ax in axes[row_idx, :]:
                    pos = ax.get_position()
                    ax.set_position(
                        [pos.x0, pos.y0 - model_idx * extra_space, pos.width, pos.height]
                    )

        # Align title with left edge of first image
        top_ax = axes[row_start, 0]
        bbox = top_ax.get_position()
        x_position = bbox.x0
        y_position = bbox.y1 + 0.01

        fig.text(
            x=x_position,
            y=y_position,
            s=model_name,
            fontdict=fontdict,
            ha="left",
            va="bottom",
        )

        # Plot each stage with color coding
        for row_idx, stage_list in enumerate(input_lists):
            row_position = row_start + row_idx
            color = "green" if row_idx == 0 else "purple"
            for col_idx, svg_code in enumerate(stage_list):
                ax = axes[row_position, col_idx]
                # Replace stroke color in SVG
                svg_colored = svg_code.replace('stroke="black"', f'stroke="{color}"')
                img = svg_to_image(svg_colored)
                ax.imshow(img)
                ax.axis("off")

    plt.savefig("fig2.png", dpi=300, bbox_inches="tight")


def main():
    # Each model: 2 inputs (square->circle morphs)
    models = {
        "SVG AE": [
            [  # Input 1 stages
                """<svg width="64" height="64"><rect width="64" height="64" rx="0" ry="0" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="4" ry="4" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="9" ry="9" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="13" ry="13" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="18" ry="18" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="22" ry="22" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="27" ry="27" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="32" ry="32" stroke="black" fill="none"/></svg>""",
            ],
            [  # Input 2 stages
                """<svg width="64" height="64"><polygon points="32,0 64,32 32,64 0,32" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="28,0 64,24 36,64 0,40" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="24,0 64,16 40,64 0,48" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="20,0 64,8 44,64 0,56" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="16,0 64,0 48,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="12,0 64,0 52,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="8,0 64,0 56,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="0,0 64,0 64,64 0,64" fill="none" stroke="black"/></svg>""",
            ],
        ],
        "MAE": [
            [  # Input 1 stages
                """<svg width="64" height="64"><rect width="64" height="64" rx="0" ry="0" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="4" ry="4" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="9" ry="9" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="13" ry="13" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="18" ry="18" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="22" ry="22" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="27" ry="27" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="32" ry="32" stroke="black" fill="none"/></svg>""",
            ],
            [  # Input 2 stages
                """<svg width="64" height="64"><polygon points="32,0 64,32 32,64 0,32" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="28,0 64,24 36,64 0,40" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="24,0 64,16 40,64 0,48" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="20,0 64,8 44,64 0,56" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="16,0 64,0 48,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="12,0 64,0 52,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="8,0 64,0 56,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="0,0 64,0 64,64 0,64" fill="none" stroke="black"/></svg>""",
            ],
        ],
        "JEPA": [
            [  # Input 1 stages
                """<svg width="64" height="64"><rect width="64" height="64" rx="0" ry="0" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="4" ry="4" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="9" ry="9" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="13" ry="13" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="18" ry="18" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="22" ry="22" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="27" ry="27" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="32" ry="32" stroke="black" fill="none"/></svg>""",
            ],
            [  # Input 2 stages
                """<svg width="64" height="64"><polygon points="32,0 64,32 32,64 0,32" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="28,0 64,24 36,64 0,40" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="24,0 64,16 40,64 0,48" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="20,0 64,8 44,64 0,56" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="16,0 64,0 48,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="12,0 64,0 52,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="8,0 64,0 56,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="0,0 64,0 64,64 0,64" fill="none" stroke="black"/></svg>""",
            ],
        ],
        "Contrastive": [
            [  # Input 1 stages
                """<svg width="64" height="64"><rect width="64" height="64" rx="0" ry="0" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="4" ry="4" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="9" ry="9" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="13" ry="13" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="18" ry="18" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="22" ry="22" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="27" ry="27" stroke="black" fill="none"/></svg>""",
                """<svg width="64" height="64"><rect width="64" height="64" rx="32" ry="32" stroke="black" fill="none"/></svg>""",
            ],
            [  # Input 2 stages
                """<svg width="64" height="64"><polygon points="32,0 64,32 32,64 0,32" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="28,0 64,24 36,64 0,40" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="24,0 64,16 40,64 0,48" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="20,0 64,8 44,64 0,56" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="16,0 64,0 48,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="12,0 64,0 52,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="8,0 64,0 56,64 0,64" fill="none" stroke="black"/></svg>""",
                """<svg width="64" height="64"><polygon points="0,0 64,0 64,64 0,64" fill="none" stroke="black"/></svg>""",
            ],
        ],
    }

    make_grid(models)


if __name__ == "__main__":
    main()
