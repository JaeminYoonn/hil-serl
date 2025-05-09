from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.holly_pickup_insertion.config import (
    TrainConfig as HollyPickupInsertionTrainConfig,
)
from experiments.usb_pickup_insertion.config import (
    TrainConfig as USBPickupInsertionTrainConfig,
)
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig

CONFIG_MAPPING = {
    "ram_insertion": RAMInsertionTrainConfig,
    "holly_pickup_insertion": HollyPickupInsertionTrainConfig,
    "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    "object_handover": ObjectHandoverTrainConfig,
    "egg_flip": EggFlipTrainConfig,
}
