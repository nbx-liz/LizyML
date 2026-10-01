"""Probe: config_version over every entry path, and how a pydantic validator behaves."""

import warnings

from pydantic import BaseModel, field_validator

from lizyml.config.loader import load_config
from lizyml.config.schema import LizyMLConfig
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model

warnings.filterwarnings("ignore")

BASE = {
    "task": "binary", "data": {"target": "y"},
    "split": {"method": "kfold", "n_splits": 2},
    "model": {"name": "lgbm"}, "evaluation": {"metrics": ["auc"]},
}


def run(label, fn):
    try:
        r = fn()
        print(f"{label:55s} -> accepted (config_version={getattr(getattr(r, '_cfg', r), 'config_version', r)!r})")
    except LizyMLError as e:
        print(f"{label:55s} -> LizyMLError {e.code.value}")
    except Exception as e:  # noqa: BLE001
        print(f"{label:55s} -> {type(e).__name__}: {str(e).splitlines()[0][:80]}")


for v in [1, 2, 0, -1, "1", "2", True, False, 1.0, 2.0, 1.5, None, "x"]:
    raw = {"config_version": v, **BASE}
    run(f"load_config(dict v={v!r})", lambda: load_config(dict(raw)))
    run(f"LizyMLConfig.model_validate(v={v!r})", lambda: LizyMLConfig.model_validate(dict(raw)))
    run(f"Model(dict v={v!r})", lambda: Model(dict(raw)))
raw = {"config_version": 2, **BASE}
run("model_construct(v=2) then Model(instance)", lambda: Model(LizyMLConfig.model_construct(**raw)))
cfg = LizyMLConfig.model_validate({"config_version": 1, **BASE})
cfg.config_version = 2
run("mutated instance v=2 then Model(instance)", lambda: Model(cfg))
run("model_copy(update v=2) then Model(instance)",
    lambda: Model(LizyMLConfig.model_validate({"config_version": 1, **BASE}).model_copy(update={"config_version": 2})))


# How does pydantic v2 treat a non-ValueError exception raised in a field validator?
class Probe(BaseModel):
    v: int

    @field_validator("v")
    @classmethod
    def _check(cls, x: int) -> int:
        if x != 1:
            raise LizyMLError(ErrorCode.CONFIG_VERSION_UNSUPPORTED, "nope", context={"config_version": x})
        return x


run("pydantic validator raising LizyMLError", lambda: Probe.model_validate({"v": 2}))
import pydantic  # noqa: E402

print("pydantic", pydantic.VERSION)
