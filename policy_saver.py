"""Atomic, CPU-portable exports of the small policy networks."""
import logging
import os
from pathlib import Path
import tempfile
import torch

logger = logging.getLogger('OmphalosLogger')

class PolicySaver:
    def __init__(self, guide_network, action_policy_network, value_network, output_dir='.'):
        self.guide_network = guide_network
        self.action_policy_network = action_policy_network
        self.value_network = value_network
        self.output_dir = Path(output_dir)

    def _save(self, model, path):
        destination = Path(path)
        if not destination.is_absolute():
            destination = self.output_dir / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
        os.close(fd)
        try:
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        logger.info('模型已导出: %s', destination)

    def save_organic_model(self, model, path='organic_prototype_model.pth'):
        self._save(model, path)

    def save_policy_models(self, blueprint_path='blueprint_policy.pth', action_path='baie_action_policy.pth', value_path='baie_value_policy.pth'):
        self._save(self.guide_network, blueprint_path)
        self._save(self.action_policy_network, action_path)
        self._save(self.value_network, value_path)
