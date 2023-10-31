# Copyright 2020 DeepMind Technologies Limited.
# Copyright 2023 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main wrapper for LapNet in JAX."""

import sys

from absl import app
from absl import flags
from absl import logging
from jax.config import config as jax_config
from lapnet import base_config
from lapnet import train
from ml_collections.config_flags import config_flags
import os
os.environ['NVIDIA_TF32_OVERRIDE']="0"

logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# internal imports

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file('config', None, 'Path to config file.')


def main(_):
  cfg = FLAGS.config
  cfg = base_config.resolve(cfg)
  if cfg.use_x64:
    jax_config.update("jax_enable_x64", True)

  logging.info('System config:\n\n%s', cfg)
  train.train(cfg)


if __name__ == '__main__':
  app.run(main)
