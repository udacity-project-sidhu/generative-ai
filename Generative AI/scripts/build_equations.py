"""Render LaTeX equations to PNGs via matplotlib mathtext.

No external LaTeX install required. Outputs are written next to the report at
`Generative AI/eq_*.png` and embedded by the markdown report.

Run: python scripts/build_equations.py
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent

EQUATIONS = {
    'eq_elbo.png': (
        r'$\mathcal{L} \,=\, \mathrm{BCE}(\hat{x},\,x) \;+\; '
        r'\frac{1}{2}\sum_i \left(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right)$',
        16,
    ),
    'eq_reparam.png': (
        r'$z \,=\, \mu + \sigma \odot \varepsilon, \quad '
        r'\varepsilon \sim \mathcal{N}(0,\,I), \quad z \in \mathbb{R}^{64}$',
        13,
    ),
    'eq_prior.png': (
        r'$z \sim \mathcal{N}(0,\,I)$',
        13,
    ),
    'eq_interp.png': (
        r'$z \,=\, (1-\alpha)\,\mu_A + \alpha\,\mu_B$',
        13,
    ),
}


def render(text: str, fontsize: int, out: Path) -> None:
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, text, fontsize=fontsize)
    fig.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.05,
                transparent=False, facecolor='white')
    plt.close(fig)


for name, (text, size) in EQUATIONS.items():
    render(text, size, OUT / name)
    print(f'wrote {OUT / name}')
