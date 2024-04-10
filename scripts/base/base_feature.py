from typing import Any


class FeatureBase:
    def __init__(self, feature_name: str) -> None:
        self.feature_name: str = feature_name
    
    def __str__(self) -> str:
        return self.feature_name
    
    def __repr__(self) -> str:
        return self.feature_name.replace(' ', '_')
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
    
    def distance(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError