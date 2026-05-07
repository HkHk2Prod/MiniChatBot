"""Generic registry for config-driven instantiation.

Each subsystem (models, tokenizers, callbacks, ...) maintains its own
Registry. Concrete classes register under a string key; YAML configs
reference them by that key. Trying to register the same key twice raises
to surface accidental name collisions early.

`import_submodules` is the companion helper for the registry pattern:
in a subpackage's `__init__.py`, after defining the registry, call
`import_submodules(__name__, __path__)` to walk every sibling module
and fire its `@REGISTRY.register(...)` decorators — no need to maintain
an explicit list of imports.
"""

from __future__ import annotations

import pkgutil
from collections.abc import Iterable
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self._name = name
        self._classes: dict[str, type[T]] = {}

    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if key in self._classes:
                raise ValueError(
                    f"{self._name} registry: '{key}' already registered to "
                    f"{self._classes[key].__name__}"
                )
            self._classes[key] = cls
            return cls

        return decorator

    def __getitem__(self, key: str) -> type[T]:
        if key not in self._classes:
            raise KeyError(
                f"{self._name} registry has no entry '{key}'. "
                f"Available: {sorted(self._classes.keys())}"
            )
        return self._classes[key]

    def keys(self) -> list[str]:
        return sorted(self._classes.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._classes

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, entries={self.keys()})"

    def __str__(self) -> str:
        if not self._classes:
            return f"{self._name} registry (empty)"
        n = len(self._classes)
        header = f"{self._name} registry ({n} {'entry' if n == 1 else 'entries'}):"
        lines = [header]
        width = max(len(k) for k in self._classes)
        for key in sorted(self._classes):
            lines.append(f"  {key:<{width}}  ->  {self._classes[key].__name__}")
        return "\n".join(lines)


def import_submodules(
    package_name: str,
    package_path: Iterable[str],
    skip: Iterable[str] = ("base",),
) -> None:
    """Import every direct submodule of a package to fire registration decorators.

    Use in a subpackage's `__init__.py` after defining the registry:

        FOO_REGISTRY = Registry("foo")
        import_submodules(__name__, __path__)

    Iterates only direct children of the package (not nested subpackages'
    contents — but importing a subpackage runs its own `__init__.py`,
    which can call `import_submodules` itself). Skips modules listed in
    `skip` (default `{"base"}`) so abstract-base files don't get pulled
    in needlessly.
    """
    skip_set = set(skip)
    for _, name, _ in pkgutil.iter_modules(package_path):
        if name in skip_set:
            continue
        __import__(f"{package_name}.{name}")
