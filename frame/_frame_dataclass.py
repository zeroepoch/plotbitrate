import dataclasses


@dataclasses.dataclass
class Frame:
    __slots__ = ["time", "size", "pict_type"]
    time: float
    size: int
    pict_type: str
    
    @staticmethod
    def get_fields():
        return [f.name for f in dataclasses.fields(Frame)]
