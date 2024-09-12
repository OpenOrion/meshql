from typing import Protocol


class Preprocess(Protocol):
    def apply(self) -> "Preprocess": ...
