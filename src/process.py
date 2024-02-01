"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig
import pandas as pd

@hydra.main(config_path="../config", config_name="main", version_base=None)
def process_data(config: DictConfig):
    """Function to process the data"""

    print(f"Process data using {config.data.raw}")
    print(f"Columns used: {config.process.use_columns}")
    df=pd.read_csv(config.data.raw)
    df.to_csv(config.data.processed, index=False)


if __name__ == "__main__":
    process_data()
