"""Vendored subset of the UNiFMIR ``model`` package.

Only the pieces needed for 2D image->image inference are kept. The upstream
``model/__init__.py`` is intentionally NOT reproduced -- it pulls in
``csbdeep``/``tensorflow``/``keras``/``gradio`` and the training-only
``ProjectionUpdater``. Here we expose just the ``swinir`` module, whose only
third-party dependencies are ``torch`` and ``einops`` (the few ``timm`` layer
helpers it needs are vendored in ``model/layers.py``).
"""

from . import swinir  # noqa: F401
from .swinir import swinir as SwinIR  # noqa: F401
