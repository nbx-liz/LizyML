"""ModelPersistenceMixin — export/load methods extracted from Model facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.logging import generate_run_id, get_logger

if TYPE_CHECKING:
    import pandas as pd

    from lizyml.config.schema import LizyMLConfig
    from lizyml.core.types.fit_result import FitResult
    from lizyml.training.refit_trainer import RefitResult

_log = get_logger("model")


class ModelPersistenceMixin:
    """Mixin providing export/load methods for :class:`Model`."""

    # Attributes provided by Model — declared for type checking only.
    if TYPE_CHECKING:
        _cfg: LizyMLConfig
        _fit_result: FitResult | None
        _refit_result: RefitResult | None
        _metrics: dict[str, Any] | None
        _y: pd.Series | None
        _X: pd.DataFrame | None
        _run_dir: Path | None
        _output_dir: str | Path | None

        def _require_fit(self) -> FitResult: ...

        def _require_refit(self) -> RefitResult: ...

    def export(self, path: str | Path | None = None) -> Path:
        """Export Model artifacts to a directory.

        Saves ``fit_result.pkl``, ``refit_model.pkl``, ``metadata.json``,
        and ``analysis_context.pkl`` under *path*.  The saved model can be
        restored with :meth:`load`, including diagnostic API support.

        Path resolution (first match wins):

        1. Explicit *path* argument.
        2. ``{run_dir}/export`` when a run directory exists from ``fit``/``tune``.
        3. New run directory under ``output_dir`` if configured.
        4. Error — no destination available.

        Args:
            path: Output directory (created if absent).  Optional when
                ``output_dir`` is configured via Config or constructor.

        Returns:
            Resolved export directory path.

        Raises:
            LizyMLError with MODEL_NOT_FIT when called before ``fit``.
            LizyMLError with SERIALIZATION_FAILED on I/O errors or when
                no path can be resolved.

        Warning:
            The ``.pkl`` files use joblib/pickle.  Only load artifacts from
            trusted sources.
        """
        fit_result = self._require_fit()
        refit_result = self._require_refit()

        resolved_path = self._resolve_export_path(path)

        from lizyml.persistence.exporter import AnalysisContext
        from lizyml.persistence.exporter import export as _export

        ctx: AnalysisContext | None = None
        if self._y is not None and self._X is not None:
            ctx = AnalysisContext(y_true=self._y, X_for_explain=self._X)

        _export(
            path=resolved_path,
            fit_result=fit_result,
            refit_result=refit_result,
            config=self._cfg.model_dump(),
            task=self._cfg.task,
            analysis_context=ctx,
        )
        _log.info("event='export.done' path=%s", resolved_path)
        return resolved_path

    def _resolve_export_path(self, path: str | Path | None) -> Path:
        """Resolve the export destination directory."""
        if path is not None:
            return Path(path)
        if self._run_dir is not None:
            return Path(self._run_dir) / "export"
        if self._output_dir is not None:
            from lizyml.core.logging import setup_output_dir

            export_run_id = generate_run_id()
            self._run_dir = setup_output_dir(self._output_dir, export_run_id)
            return Path(self._run_dir) / "export"
        raise LizyMLError(
            ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                "No export path provided and no output_dir configured. "
                "Pass an explicit path or set output_dir in Config / constructor."
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> Any:
        """Restore a Model from a directory created by :meth:`export`.

        Args:
            path: Directory containing ``metadata.json``, ``fit_result.pkl``,
                and ``refit_model.pkl``.

        Returns:
            A :class:`Model` instance ready for ``predict`` and ``evaluate``.

        Raises:
            LizyMLError with DESERIALIZATION_FAILED on validation or I/O errors.

        Warning:
            Only load from trusted sources — joblib uses pickle internally.
        """
        from lizyml.persistence.loader import load as _load

        fit_result, refit_result, metadata, analysis_context = _load(path)
        config = metadata["config"]
        instance = cls(config)  # type: ignore[call-arg]  # cls is Model at runtime
        instance._fit_result = fit_result
        instance._refit_result = refit_result
        instance._metrics = fit_result.metrics
        if analysis_context is not None:
            instance._y = analysis_context.y_true
            instance._X = analysis_context.X_for_explain
        _log.info("event='load.done' path=%s run_id=%s", path, metadata.get("run_id"))
        return instance
