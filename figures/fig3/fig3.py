# file: svg_table_figure.py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cairosvg
from io import BytesIO


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
        "family": "Times New Roman",
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
    # Example SVG queries (3 per model)
    input_svgs = [
        "<svg width='64' height='64'><circle cx='32' cy='32' r='20' stroke='black' fill='none'/></svg>",
        "<svg width='64' height='64'><rect x='10' y='10' width='40' height='40' stroke='black' fill='none'/></svg>",
        "<svg width='64' height='64'><polygon points='32,10 54,54 10,54' stroke='black' fill='none'/></svg>",
    ]

    # Example unique reconstructions for each query
    reconstructions = {
        "SVG AE": [
            [
                "<svg width='64' height='64'><circle cx='30' cy='30' r='22' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='32' cy='32' r='21' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='34' cy='34' r='20' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='36' cy='36' r='19' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='38' cy='38' r='18' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><rect x='10' y='10' width='40' height='40' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='11' y='11' width='39' height='39' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='12' y='12' width='38' height='38' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='13' y='13' width='37' height='37' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='14' y='14' width='36' height='36' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><polygon points='32,10 54,54 10,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='34,10 52,54 12,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='36,10 50,54 14,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='38,10 48,54 16,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='40,10 46,54 18,54' stroke='black' fill='none'/></svg>",
            ],
        ],
        "MAE": [
            [
                "<svg width='64' height='64'><circle cx='32' cy='32' r='18' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='31' cy='31' r='19' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='30' cy='30' r='20' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='29' cy='29' r='21' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='28' cy='28' r='22' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><rect x='11' y='14' width='40' height='36' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='12' y='15' width='39' height='35' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='13' y='16' width='38' height='34' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='14' y='17' width='37' height='33' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='15' y='18' width='36' height='32' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><polygon points='32,10 54,54 10,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='32,10 52,54 12,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='32,10 50,54 14,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='32,10 48,54 16,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='32,10 46,54 18,54' stroke='black' fill='none'/></svg>",
            ],
        ],
        "JEPA": [
            [
                "<svg width='64' height='64'><circle cx='32' cy='32' r='20' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='33' cy='31' r='19' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='34' cy='30' r='18' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='35' cy='29' r='17' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='36' cy='28' r='16' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><rect x='10' y='10' width='42' height='42' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='11' y='11' width='41' height='41' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='12' y='12' width='40' height='40' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='13' y='13' width='39' height='39' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='14' y='14' width='38' height='38' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><polygon points='32,10 54,54 10,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='31,10 53,54 11,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='30,10 52,54 12,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='29,10 51,54 13,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='28,10 50,54 14,54' stroke='black' fill='none'/></svg>",
            ],
        ],
        "Contrastive": [
            [
                "<svg width='64' height='64'><circle cx='31' cy='31' r='19' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='32' cy='32' r='18' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='33' cy='33' r='17' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='34' cy='34' r='16' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><circle cx='35' cy='35' r='15' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><rect x='13' y='13' width='36' height='36' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='14' y='14' width='35' height='35' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='15' y='15' width='34' height='34' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='16' y='16' width='33' height='33' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><rect x='17' y='17' width='32' height='32' stroke='black' fill='none'/></svg>",
            ],
            [
                "<svg width='64' height='64'><polygon points='32,10 54,54 10,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='33,10 53,54 11,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='34,10 52,54 12,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='35,10 51,54 13,54' stroke='black' fill='none'/></svg>",
                "<svg width='64' height='64'><polygon points='36,10 50,54 14,54' stroke='black' fill='none'/></svg>",
            ],
        ],
    }

    rows = []
    models_queries_count = {}

    # Convert SVGs to in-memory PNGs and store them
    for model, model_queries in reconstructions.items():
        models_queries_count[model] = len(model_queries)
        for q_idx, svg_code in enumerate(input_svgs):
            # Query image in-memory
            query_buf = BytesIO()
            cairosvg.svg2png(bytestring=svg_code.encode("utf-8"), write_to=query_buf)
            query_buf.seek(0)

            # Retrieval images in-memory (unique per query)
            retrieval_bufs = []
            for r_svg in model_queries[q_idx]:
                r_buf = BytesIO()
                cairosvg.svg2png(bytestring=r_svg.encode("utf-8"), write_to=r_buf)
                r_buf.seek(0)
                retrieval_bufs.append(r_buf)

            rows.append({"model": model, "query": query_buf, "retrievals": retrieval_bufs})

    # Draw figure
    make_figure(rows, models_queries_count)


if __name__ == "__main__":
    main()
