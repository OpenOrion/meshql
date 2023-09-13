from typing import Iterable, Iterator, MutableSet, TypeVar, Union
import numpy as np

NumpyFloat = np.float32
Number = Union[int, float, NumpyFloat]
VectorTuple = Union[tuple[float, float, float], tuple[float, float]]
LineTuple = tuple[VectorTuple, VectorTuple]

T = TypeVar("T")
class OrderedSet(MutableSet[T]):
    """A set that preserves insertion order by internally using a dict."""

    def __init__(self, iterable: Iterable[T] = []):
        self._d = dict.fromkeys(iterable)

    def add(self, x: T) -> None:
        self._d[x] = None

    def discard(self, x: T) -> None:
        self._d.pop(x, None)

    def update(self, iterable: Iterable[T]) -> None:
        self._d.update(dict.fromkeys(iterable))

    @property
    def first(self) -> T:
        return next(iter(self))

    @property
    def last(self) -> T:
        *_, last = iter(self)
        return last

    def __contains__(self, x: object) -> bool:
        return self._d.__contains__(x)

    def __len__(self) -> int:
        return self._d.__len__()

    def __iter__(self) -> Iterator[T]:
        return self._d.__iter__()

    def __str__(self):
        return f"{{{', '.join(str(i) for i in self)}}}"

    def __repr__(self):
        return f"<OrderedSet {self}>"

    # def __getitem__(self, index):
    #     return list(self)[index]
