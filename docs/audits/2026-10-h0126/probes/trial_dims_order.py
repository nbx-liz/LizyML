"""Trial: recorded_params returns parameters in the round's dimension order."""

p = "lizyml/tuning/_step_grid.py"
s = open(p, encoding="utf-8").read()
old = '''    by_name = {dim.name: dim for dim in dims}
    out: dict[str, Any] = {}
    for name, value in params.items():
        dim = by_name.get(name)
        if dim is None:
            raise ValueError(
                f"trial parameter {name!r} has no dimension in its round's space "
                f"{sorted(by_name)}"
            )
        if isinstance(dim, CategoricalDim) or dim.step is None:
            out[name] = value
        else:
            out[name] = grid_value(dim, value)
    return out'''
new = '''    by_name = {dim.name: dim for dim in dims}
    for name in params:
        if name not in by_name:
            raise ValueError(
                f"trial parameter {name!r} has no dimension in its round's space "
                f"{sorted(by_name)}"
            )
    out: dict[str, Any] = {}
    for dim in dims:
        if dim.name not in params:
            continue
        value = params[dim.name]
        if isinstance(dim, CategoricalDim) or dim.step is None:
            out[dim.name] = value
        else:
            out[dim.name] = grid_value(dim, value)
    return out'''
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched")
