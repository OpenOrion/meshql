from typing import Iterable, Iterator, Literal, MutableSet, TypeVar, Union
import numpy as np
import cadquery as cq
NumpyFloat = np.float32
Number = Union[int, float, NumpyFloat]
VectorTuple = Union[tuple[Number, Number, Number], tuple[Number, Number], np.ndarray]
VectorLike = Union[VectorTuple, cq.Vector]
LineTuple = tuple[VectorTuple, VectorTuple]
Axis = Union[Literal["X", "Y", "Z"], VectorTuple, cq.Vector]

def to_array(vec: VectorLike):
    if isinstance(vec, np.ndarray):
        return vec
    elif isinstance(vec, tuple):
        return np.array(vec if len(vec) == 3 else (*vec, 0))
    elif isinstance(vec, cq.Vector):
        return np.array(vec.toTuple())

def to_2d_array(vecs: Iterable[VectorLike]):
    return [to_array(vec) for vec in vecs]

def to_vec(axis: Axis):
    if isinstance(axis, str):
        vec = cq.Vector([1 if axis == "X" else 0, 1 if axis == "Y" else 0, 1 if axis == "Z" else 0])        
    elif isinstance(axis, tuple):
        vec = cq.Vector(axis)
    elif isinstance(axis, np.ndarray):
        vec = cq.Vector(tuple(axis))
    else:
        vec = axis
    return vec


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

    def difference(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy -= d
        return copy

    def union(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy |= d
        return copy

    def intersection(self, d: "OrderedSet") -> "OrderedSet[T]":
        copy = OrderedSet(self)
        copy &= d
        return copy

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
