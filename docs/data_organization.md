## Data organization

Each dataset must follow the structure:

root_data/
└── dataset_name/
    ├── dataset_info.txt
    ├── dump/
    └── preprocess/
        ├── raw_data/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── recon_data/
            ├── train/
            ├── val/
            └── test/

src/
└── lisai/
    └── data/
        └── data_prep
    └── lib/
        └── hdn/
        └── upsamp/
        └── utils/
            └── data_utils.py
            └── misc.py