"""
Extract embeddings from Moirai2 (decoder-only, causal attention) via forward hook.

Architecture note (confirmed from HuggingFace model card + source):
  Moirai2 is DECODER-ONLY with causal temporal attention (not BERT-style encoder).
  For context_length=200, prediction_length=1 the transformer processes 14 patches:
    patches 0-12 : context (each patch_size=16 days of log-returns)
    patch 13     : MASK/prediction token (1 forecast step)

  Causal attention: patch t attends to patches 0..t only.
  The MASK token (patch -1) has attended to ALL 13 context patches
  and is therefore the richest single-vector representation.

Pooling strategies (controlled by `pooling` param):
  'last'         : reprs[:, -1, :]          ← MASK token; sees all context  [DEFAULT]
  'last_context' : reprs[:, -2, :]          ← last OBSERVED patch; sees all context
  'mean'         : reprs.mean(dim=1)         ← old behaviour; diluted by early patches
  'mean_context' : reprs[:, :-1, :].mean(1) ← mean of context patches only (excl. MASK)

Usage:
    python src/embed_extractor.py
"""
import warnings
warnings.filterwarnings("ignore", message="The mean prediction is not stored")

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# Valid pooling strategies
_POOLING_OPTIONS = {"last", "last_context", "mean", "mean_context"}


class Moirai2Embedder:
    """
    Wraps Moirai2Forecast and extracts hidden states via a forward hook on
    Moirai2Module.encoder (the causal transformer stack).

    Hook output: (batch, n_patches, d_model=384)
    Returned embedding: (d_model,) per series after applying `pooling`.
    """

    D_MODEL = 384  # Moirai2-small d_model (from config.json)

    def __init__(
        self,
        size: str = "small",
        context_length: int = 200,
        patch_size: int = 32,       # kept for API compatibility; actual patch size is set by the pretrained model (16)
        device: str | None = None,
        pooling: str = "last",      # see module docstring for options
    ):
        if pooling not in _POOLING_OPTIONS:
            raise ValueError(f"pooling must be one of {_POOLING_OPTIONS}, got {pooling!r}")
        self.context_length = context_length
        self.patch_size = patch_size
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading Moirai2-{size} on {self.device}...")
        self._module = Moirai2Module.from_pretrained(
            f"Salesforce/moirai-2.0-R-{size}"
        ).to(self.device)
        self._module.eval()

        self._captured: dict = {}
        self._hook_handle = None

    def _register_hook(self):
        def hook_fn(layer, inp, output):
            # output: (batch, n_patches, d_model)
            self._captured["reprs"] = output.detach().cpu()

        self._hook_handle = self._module.encoder.register_forward_hook(hook_fn)

    def _remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def embed_series(
        self,
        series_dict: dict[str, pd.Series],
        prediction_length: int = 1,
    ) -> dict[str, np.ndarray]:
        """
        Extract embeddings for each time series.

        Args:
            series_dict: {name: pd.Series with DatetimeIndex}
            prediction_length: dummy forecast horizon (does not affect embeddings)

        Returns:
            {name: np.ndarray of shape (d_model,)} — one embedding per series
        """
        names = list(series_dict.keys())
        df = pd.DataFrame(series_dict)

        ds = PandasDataset(dict(df))

        # Build a 1-window test split using the last context_length points
        _, test_template = split(ds, offset=-prediction_length)
        test_data = test_template.generate_instances(
            prediction_length=prediction_length,
            windows=1,
            distance=prediction_length,
        )
        inputs = list(test_data.input)

        # Build Moirai2Forecast with dummy dims
        forecast_model = Moirai2Forecast(
            module=self._module,
            prediction_length=prediction_length,
            context_length=self.context_length,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
        )
        predictor = forecast_model.create_predictor(batch_size=len(names))

        self._register_hook()
        try:
            with torch.no_grad():
                _ = list(predictor.predict(iter(inputs)))
        finally:
            self._remove_hook()

        reprs = self._captured.get("reprs")  # (batch, n_patches, 384)
        if reprs is None:
            raise RuntimeError("Hook did not capture any output.")

        # Pool over patch dimension → (batch, 384)
        if self.pooling == "last":
            # MASK/prediction token — has attended to ALL context patches
            embeddings = reprs[:, -1, :].numpy()
        elif self.pooling == "last_context":
            # Last observed context patch (index -2, before the MASK token)
            embeddings = reprs[:, -2, :].numpy()
        elif self.pooling == "mean_context":
            # Mean of context patches only, excluding the MASK token
            embeddings = reprs[:, :-1, :].mean(dim=1).numpy()
        else:
            # 'mean': original behaviour — average over all patches incl. MASK
            embeddings = reprs.mean(dim=1).numpy()

        # The predictor may reorder or batch differently; align by index
        # (for 1 window per series, batch dim = n_series in order)
        return {name: embeddings[i] for i, name in enumerate(names)}

    def embed_windows(
        self,
        series: pd.Series,
        window_size: int,
        stride: int | None = None,
        prediction_length: int = 1,
    ) -> np.ndarray:
        """
        Sliding-window embedding for a single series.
        Useful for detecting regime changes over time.

        Returns:
            np.ndarray of shape (n_windows, d_model)
        """
        stride = stride or window_size
        values = series.values
        n = len(values)
        windows = []
        starts = []

        for start in range(0, n - window_size - prediction_length + 1, stride):
            windows.append(values[start : start + window_size])
            starts.append(series.index[start])

        if not windows:
            raise ValueError(f"Series too short for window_size={window_size}")

        all_embeddings = []
        for i, (win, ts_start) in enumerate(zip(windows, starts)):
            s = pd.Series(win, index=pd.date_range(ts_start, periods=len(win), freq=series.index.freq or "B"))
            result = self.embed_series({"series": s}, prediction_length=prediction_length)
            all_embeddings.append(result["series"])

        return np.stack(all_embeddings)  # (n_windows, 384)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA

    os.makedirs("results", exist_ok=True)

    # 1. Load demo dataset
    print("Loading demo dataset (10 series)...")
    url = (
        "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
        "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
    )
    df = pd.read_csv(url, index_col=0, parse_dates=True)
    print(f"  Shape: {df.shape} | Columns: {list(df.columns)}")

    # 2. Extract embeddings
    embedder = Moirai2Embedder(size="small", context_length=200)
    series_dict = {col: df[col].dropna() for col in df.columns}

    print(f"\nExtracting embeddings for {len(series_dict)} series...")
    embeddings = embedder.embed_series(series_dict, prediction_length=20)

    # 3. Print stats
    print(f"\n=== Embedding Stats ===")
    print(f"  Shape per series : ({Moirai2Embedder.D_MODEL},)")
    for name, emb in embeddings.items():
        print(f"  {name:6s}  mean={emb.mean():.4f}  std={emb.std():.4f}"
              f"  min={emb.min():.4f}  max={emb.max():.4f}")

    # 4. Pairwise cosine similarity
    names = list(embeddings.keys())
    emb_matrix = np.stack([embeddings[n] for n in names])  # (n, 384)
    cos_sim = cosine_similarity(emb_matrix)

    print(f"\n=== Cosine Similarity Matrix ===")
    header = f"{'':8s}" + "".join(f"{n:8s}" for n in names)
    print(header)
    for i, ni in enumerate(names):
        row = f"{ni:8s}" + "".join(f"{cos_sim[i,j]:8.3f}" for j in range(len(names)))
        print(row)

    # 5. PCA visualization (2D)
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(emb_matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=100, color="steelblue", zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (emb_2d[i, 0], emb_2d[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title(f"Moirai2 Embeddings — PCA 2D\n"
                 f"(explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/embeddings_pca.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: results/embeddings_pca.png")

    # 6. Sliding-window embedding for 1 series (regime visualization)
    print(f"\nSliding-window embedding for series '{names[0]}'...")
    first_series = df[names[0]].dropna()
    first_series.index = pd.DatetimeIndex(first_series.index)
    if not first_series.index.freq:
        first_series = first_series.asfreq("h", method="ffill")

    win_embeddings = embedder.embed_windows(
        first_series, window_size=100, stride=20, prediction_length=10
    )
    print(f"  Sliding-window shape: {win_embeddings.shape}")

    # Project to 1D via PCA to see temporal evolution
    pca1 = PCA(n_components=1)
    emb_1d = pca1.fit_transform(win_embeddings).flatten()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    axes[0].plot(first_series.values, linewidth=0.8, color="steelblue")
    axes[0].set_title(f"Series '{names[0]}' — raw values")
    axes[1].plot(emb_1d, color="tomato", marker="o", markersize=3)
    axes[1].set_title("Moirai2 embedding PC1 over time (regime signal)")
    axes[1].set_xlabel("Window index")
    plt.tight_layout()
    plt.savefig("results/embeddings_temporal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/embeddings_temporal.png")
    print("\nDone.")
