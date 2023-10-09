from abc import (
    ABC as abstract_class,
    abstractmethod as abstract_method)
from dataclasses import dataclass
from typing import Any, Optional

class meta_sample:
    def __init__(self, identity):
        self.identity = identity
    def __hash__(self):
        return self.identity

class meta_ds(abstract_class):
    @abstract_method
    def __getitem__(self, ind: int) -> meta_sample:
        raise NotImplementedError()
    @abstract_method
    def __len__(self):
        raise NotImplementedError()

@dataclass
class modeling_sample:
    sample : meta_sample
    x : Optional[Any] = None
    y : Optional[Any] = None
    y_hat : Optional[Any] = None
    def __hash__(self):
        return hash(self.sample)


class modeling_ds:
    def __init__(self, meta_ds, x_creator,
            y_creator = None):
        self.x_creator = x_creator
        self.y_creator = y_creator
        self.meta_ds = meta_ds
    def __len__(self):
        return len(self.meta_ds)
    def __getitem__(self, index) -> modeling_sample:
        curr_meta_data_sample = self.meta_ds[index]
        ret_sample = modeling_sample(curr_meta_data_sample)
        ret_sample.x = self.x_creator(curr_meta_data_sample)
        if self.y_creator:
            ret_sample.y = self.y_creator(curr_meta_data_sample)
        else:
            ret_sample.y = None
        ret_sample.sample = curr_meta_data_sample
        return ret_sample

class epoch_training_meter (abstract_class):
    @abstract_method
    def __init__(self, display_name, *args, **kwds):
        self.display_name = display_name
        self.reset()
    @abstract_method
    def reset (self):
        pass
    @abstract_method
    def __call__(self, y_hat, y, meta_data_samples,\
        *args, **kwds):
        pass
    @abstract_method
    def display(self):
        pass