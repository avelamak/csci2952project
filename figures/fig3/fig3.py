# file: svg_table_figure.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cairosvg
from io import BytesIO

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


def make_figure(rows, models_queries_count):
    num_retrievals = 5
    n_rows = len(rows)
    n_cols = 1 + 1 + num_retrievals  # Model + Query + 5 retrievals

    fig, ax = plt.subplots(figsize=(10, 2 + n_rows))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cell_width = 1 / n_cols
    cell_height = 1 / (n_rows + 1)

    fontdict = {
        "fontsize": 18,
        "fontweight": "normal",
        "family": "serif",
        "color": "black",
    }

    # Headers
    ax.text(
        cell_width / 2, 1 - cell_height / 2, "Model", ha="center", va="center", fontdict=fontdict
    )
    ax.text(
        cell_width + cell_width / 2,
        1 - cell_height / 2,
        "Query",
        ha="center",
        va="center",
        fontdict=fontdict,
    )
    ax.text(
        2 * cell_width + (num_retrievals * cell_width) / 2,
        1 - cell_height / 2,
        f"Top {num_retrievals} retrieval results on all glyphs",
        ha="center",
        va="center",
        fontdict=fontdict,
    )

    # Horizontal line under header
    ax.hlines(1 - cell_height, 0, 1, colors="black", linewidth=1)
    # Vertical line after Query column
    ax.vlines(2 * cell_width, 0, 1, colors="black", linewidth=1)

    # Keep track of which row each model starts
    model_start_idx = {}
    for idx, row in enumerate(rows):
        if row["model"] not in model_start_idx:
            model_start_idx[row["model"]] = idx

    # Draw rows
    for i, row in enumerate(rows):
        # Model label only for first query row
        if i == model_start_idx[row["model"]]:
            num_rows_for_model = models_queries_count[row["model"]]
            y_top = 1 - (i + 1) * cell_height
            y_bottom = 1 - (i + 1 + num_rows_for_model) * cell_height
            ax.text(
                cell_width / 2,
                (y_top + y_bottom) / 2,
                row["model"],
                ha="center",
                va="center",
                fontdict=fontdict,
            )

        # Query image
        try:
            img = mpimg.imread(row["query"])
            ax.imshow(
                img,
                extent=[
                    cell_width,
                    2 * cell_width,
                    1 - (i + 2) * cell_height,
                    1 - (i + 1) * cell_height,
                ],
                aspect="auto",
                origin="upper",
            )
        except Exception:
            ax.text(
                cell_width + cell_width / 2,
                1 - (i + 1.5) * cell_height,
                "Query",
                ha="center",
                va="center",
                color="red",
            )

        # Retrieval images (top 5 only)
        for j, r_img in enumerate(row["retrievals"][:num_retrievals]):  # <-- limit here
            x0 = (j + 2) * cell_width
            x1 = x0 + cell_width
            try:
                img = mpimg.imread(r_img)
                ax.imshow(
                    img,
                    extent=[x0, x1, 1 - (i + 2) * cell_height, 1 - (i + 1) * cell_height],
                    aspect="auto",
                    origin="upper",
                )
            except Exception:
                ax.text(
                    (x0 + x1) / 2,
                    1 - (i + 1.5) * cell_height,
                    f"R{j + 1}",
                    ha="center",
                    va="center",
                    color="red",
                )

    # Horizontal lines after last query of each model
    idx = 0
    model_names = list(models_queries_count.keys())
    for model_idx, (model, count) in enumerate(models_queries_count.items()):
        if model_idx == len(model_names) - 1:
            break
        last_row_idx = idx + count - 1
        y = 1 - (last_row_idx + 2) * cell_height
        ax.hlines(y, 0, 1, colors="black", linewidth=1)
        idx += count

    plt.savefig("fig3.png", dpi=300, bbox_inches="tight")


def main():
    import os

    base_dir = "/oscar/scratch/zzhan215/retrieval/"

    # --- EDIT THIS LIST TO ADD ANY NUMBER OF FOLDERS ---
    folder_names = [
        "AbhayaLibre-Medium_upper_Z",
        "Gabriela-Regular_6",
        "SpaceMono-BoldItalic_lower_e",
    ]

    # Map directory names → pretty model names
    model_map = {
        "svgvae": "SVG AE",
        "multimae": "MAE",
        "jepa": "JEPA",
        "contrastive": "Contrastive",
    }

    models = ["svgvae", "multimae", "jepa", "contrastive"]

    rows = []
    models_queries_count = {model_map[m]: 0 for m in models}

    for folder_name in folder_names:
        for model in models:
            model_dir = os.path.join(base_dir, model, folder_name)

            gt_path = os.path.join(model_dir, "gt.svg")
            retrieval_paths = [os.path.join(model_dir, f"rank{i}.svg") for i in range(1, 6)]

            if not os.path.exists(gt_path):
                continue

            # ---- Load GT (query) ----
            with open(gt_path, "r") as f:
                gt_svg = f.read()

            gt_svg = thicken_svg(gt_svg, factor=6.0)

            query_buf = BytesIO()
            cairosvg.svg2png(bytestring=gt_svg.encode("utf-8"), write_to=query_buf)
            query_buf.seek(0)

            # ---- Load retrievals ----
            retrieval_bufs = []
            for r_path in retrieval_paths:
                if not os.path.exists(r_path):
                    continue

                with open(r_path, "r") as f:
                    r_svg = f.read()

                r_svg = thicken_svg(r_svg, factor=6.0)

                r_buf = BytesIO()
                cairosvg.svg2png(bytestring=r_svg.encode("utf-8"), write_to=r_buf)
                r_buf.seek(0)
                retrieval_bufs.append(r_buf)

            pretty_name = model_map.get(model, model)

            rows.append({"model": pretty_name, "query": query_buf, "retrievals": retrieval_bufs})

            models_queries_count[pretty_name] += 1
    # Reorder rows so that each model's queries appear contiguously
    grouped = {model_map[m]: [] for m in models}

    for r in rows:
        grouped[r["model"]].append(r)

    # Flatten the groups in the order: svgvae, multimae, jepa, contrastive
    ordered_rows = []
    for m in models:
        pretty = model_map[m]
        ordered_rows.extend(grouped[pretty])

    rows = ordered_rows

    # Draw the stacked figure
    make_figure(rows, models_queries_count)


if __name__ == "__main__":
    main()
