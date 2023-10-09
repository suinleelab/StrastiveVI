from typing import NamedTuple


class _REGISTRY_KEYS_NT(NamedTuple):
    X_KEY: str = "X"
    BATCH_KEY: str = "batch"
    LABELS_KEY: str = "labels"
    PROTEIN_EXP_KEY: str = "proteins"
    CAT_COVS_KEY: str = "extra_categorical_covs"
    CONT_COVS_KEY: str = "extra_continuous_covs"
    INDICES_KEY: str = "ind_x"
    SIZE_FACTOR_KEY: str = "size_factor"
    LATENT_MODE_KEY: str = "latent_mode"
    LATENT_QZM_KEY: str = "latent_qzm"
    LATENT_QZV_KEY: str = "latent_qzv"
    CONT_PHENOS_KEY: str = "cont_phenos"
    CAT_PHENOS_KEY: str = "cat_phenos"
    CONT_BACKS_KEY: str = "cont_backs"
    CAT_BACKS_KEY: str = "cat_backs"



REGISTRY_KEYS = _REGISTRY_KEYS_NT()