from PIL import Image
import matplotlib.pyplot as plt
import io
import cairosvg
from pathlib import Path

import xml.etree.ElementTree as ET


def thicken_svg(svg_string, factor=2.0):
    try:
        root = ET.fromstring(svg_string)

        # namespaces fix
        ET.register_namespace("", "http://www.w3.org/2000/svg")

        for elem in root.iter():
            # Process explicit stroke-width attributes
            if "stroke-width" in elem.attrib:
                try:
                    w = float(elem.attrib["stroke-width"])
                    elem.attrib["stroke-width"] = str(w * factor)
                except ValueError:
                    pass

            # Sometimes stroke styles are inside style="stroke-width: X;"
            if "style" in elem.attrib:
                style = elem.attrib["style"]
                parts = style.split(";")
                new_parts = []
                changed = False

                for p in parts:
                    if "stroke-width" in p:
                        try:
                            key, val = p.split(":")
                            w = float(val)
                            new_parts.append(f"{key}:{w * factor}")
                            changed = True
                        except (ValueError, TypeError):
                            new_parts.append(p)
                    else:
                        new_parts.append(p)

                if changed:
                    elem.attrib["style"] = ";".join(new_parts)

            # If element has a stroke but no width, set default thick width
            elif "stroke" in elem.attrib and "stroke-width" not in elem.attrib:
                elem.attrib["stroke-width"] = str(1.0 * factor)

        return ET.tostring(root, encoding="unicode")

    except ET.ParseError:
        # If SVG parsing fails, return original
        return svg_string


def load_svg_strings(svg_dir, filenames):
    """Load SVG files and return their contents as strings."""
    svgs = []
    for name in filenames:
        path = Path(svg_dir) / name
        with open(path, "r") as f:
            svgs.append(f.read())
    return svgs


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
        "family": "serif",
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
                # Thicken SVG
                svg_thick = thicken_svg(svg_colored, factor=6.0)
                img = svg_to_image(svg_thick)
                ax.imshow(img)
                ax.axis("off")

    plt.savefig("fig2.png", dpi=300, bbox_inches="tight")


def main():
    svg_ae_dir1 = "SVGAE/interp_output_b_to_o"
    svg_ae_dir2 = "SVGAE/interp_output_s_to_7"

    svg_filenames = [
        "interp_a.svg",
        "interp_00_t0.00.svg",
        "interp_01_t0.20.svg",
        "interp_02_t0.40.svg",
        "interp_03_t0.60.svg",
        "interp_04_t0.80.svg",
        "interp_05_t1.00.svg",
        "interp_b.svg",
    ]

    models = {
        "SVG AE": [
            load_svg_strings(svg_ae_dir1, svg_filenames),
            load_svg_strings(svg_ae_dir2, svg_filenames),
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
