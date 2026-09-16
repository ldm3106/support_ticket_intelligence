"""Create and execute the concise exploration notebook with the active environment."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
notebook = nbformat.v4.new_notebook()
notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
notebook.cells = [
    nbformat.v4.new_markdown_cell("# BANKING77 text exploration\n\nCustomer-service intent classification. Source: [PolyAI](https://huggingface.co/datasets/PolyAI/banking77), CC BY 4.0; Casanueva et al. (2020). Inspect development data for modeling decisions; keep official test for final evaluation."),
    nbformat.v4.new_code_cell("from support_ticket.eda import run\nfrom support_ticket.data import load_prepared\nfrom support_ticket.config import REPORTS\nimport pandas as pd\nfrom IPython.display import display, Image\nsummary = run()\nframes, labels = load_prepared()\nsummary"),
    nbformat.v4.new_code_cell("pd.read_csv(REPORTS / 'class_distribution.csv', index_col=0).describe()"),
    nbformat.v4.new_code_cell("display(frames['train'][['text', 'label']].sample(8, random_state=42))\nlengths = frames['train'].text.str.split().str.len()\ndisplay(frames['train'].loc[lengths.nsmallest(3).index, ['text', 'label']])\ndisplay(frames['train'].loc[lengths.nlargest(3).index, ['text', 'label']])"),
    nbformat.v4.new_code_cell("display(Image(filename=str(REPORTS / 'figures' / 'text_eda.png')))"),
    nbformat.v4.new_markdown_cell("## Interpretation\n\nQueries are short, but lengths vary. Training supports vary by intent; the official test has 40 queries per intent. Macro F1 makes class-level weaknesses visible even when headline accuracy is strong. Normalized duplicates and train/test overlaps are handled before splitting. Exact deduplication does not eliminate paraphrase leakage. Full-length helpdesk conversations and non-banking requests remain outside this benchmark's scope."),
]
NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
path = root / "notebooks" / "01_data_exploration.ipynb"
path.parent.mkdir(exist_ok=True)
nbformat.write(notebook, path)
print(path)
