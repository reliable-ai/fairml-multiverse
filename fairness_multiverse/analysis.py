from typing import Optional
import pandas as pd

from pathlib import Path
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.preprocessing import OrdinalEncoder

from fanova import fANOVA


class MultiverseFanova:
    def __init__(self, features: pd.DataFrame, outcome: pd.Series) -> None:
        self.fanova_features = self.process_features(features)
        self.fanova_outcome = outcome

        self.configuration_space = self.generate_configuration_space()
        self.fanova = self.init_fanova()

    def process_features(self, features: pd.DataFrame):
        # Encode ordinally
        return pd.DataFrame(
            OrdinalEncoder().fit(features).transform(features), columns=features.columns
        )

    def generate_configuration_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for colname in self.fanova_features.columns:
            col = self.fanova_features[colname]
            cs.add_hyperparameter(
                CategoricalHyperparameter(name=colname, choices=list(col.unique()))
            )
        return cs

    def init_fanova(self):
        return fANOVA(
            self.fanova_features,
            self.fanova_outcome,
            config_space=self.configuration_space,
        )

    def quantify_importance(self, save_to: Optional[Path] = None):
        param_list = [
            param.name for param in self.configuration_space.get_hyperparameters()
        ]
        fanova_all_effects = self.fanova.quantify_importance(param_list)
        df_importance_interactions = (
            pd.DataFrame(fanova_all_effects).transpose().reset_index(drop=False)
        )
        df_importance_interactions.sort_values(
            by="individual importance", ascending=False, inplace=True
        )

        if save_to is not None:
            df_importance_interactions.to_csv(save_to)

        return df_importance_interactions

    def quantify_individual_importance(self, save_to: Optional[Path] = None):
        param_list = [
            param.name for param in self.configuration_space.get_hyperparameters()
        ]

        main_effects = {}
        for param in param_list:
            param_fanova = (param,)
            main_effects[param] = self.fanova.quantify_importance(param_fanova)[param_fanova]

        df_main_effects = pd.DataFrame(main_effects).transpose()
        df_main_effects.sort_values(by="individual importance", ascending=False)

        if save_to is not None:
            df_main_effects.to_csv(df_main_effects)

        return df_main_effects
