"""
This module contains wrapper functions to analyze the output from a multiverse
analysis esp. in regards to conducting a FANOVA.
"""

from typing import Optional
import pandas as pd

from pathlib import Path
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from sklearn.preprocessing import OrdinalEncoder

from fanova import fANOVA


class MultiverseFanova:
    def __init__(self, features: pd.DataFrame, outcome: pd.Series) -> None:
        """
        Initializes a new MultiverseFanova instance.

        MiltiverseFanova is a helper class to perform a fANOVA analysis on
        data from a multiverse analysis.

        Args:
        - features: A pandas DataFrame containing the features / decisions to
            be used in the fANOVA analysis.
        - outcome: A pandas Series containing the outcome variable to be
            used in the fANOVA analysis.
        """
        self.fanova_features = self.process_features(features)
        self.fanova_outcome = outcome

        self.configuration_space = self.generate_configuration_space()
        self.fanova = self.init_fanova()

    def process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses a set of features for a fANOVA analysis.

        Args:
        - features: A pandas DataFrame containing the features to be encoded.

        Returns:
        - A pandas DataFrame containing the encoded features.
        """
        return pd.DataFrame(
            OrdinalEncoder().fit(features).transform(features), columns=features.columns
        )

    def generate_configuration_space(self) -> ConfigurationSpace:
        """
        Generates a ConfigurationSpace object based on the instance features.

        Returns:
        - A ConfigurationSpace object.
        """
        cs = ConfigurationSpace()
        for colname in self.fanova_features.columns:
            col = self.fanova_features[colname]
            cs.add_hyperparameter(
                CategoricalHyperparameter(name=colname, choices=list(col.unique()))
            )
        return cs

    def init_fanova(self):
        """
        Initializes a fanova.fANOVA object.

        This will not yet run the analysis to compute importance measures.

        Returns:
        - An fANOVA object.
        """
        return fANOVA(
            self.fanova_features,
            self.fanova_outcome,
            config_space=self.configuration_space,
        )

    def quantify_importance(self, save_to: Optional[Path] = None):
        """
        Quantifies the joint importance of features in the MultiverseFanova.

        Args:
        - save_to: A Path specifying where to save the results. (optional)

        Returns:
        - A pandas DataFrame containing the importance of each feature.
        """
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
        """
        Quantifies the individual importance of each feature in the MultiverseFanova.

        Args:
        - save_to: A Path specifying where to save the results. (optional)

        Returns:
        - A pandas DataFrame containing the individual importance of each feature.
        """
        param_list = [
            param.name for param in self.configuration_space.get_hyperparameters()
        ]

        main_effects = {}
        for param in param_list:
            param_fanova = (param,)
            main_effects[param] = self.fanova.quantify_importance(param_fanova)[
                param_fanova
            ]

        df_main_effects = pd.DataFrame(main_effects).transpose()
        df_main_effects.sort_values(by="individual importance", ascending=False)

        if save_to is not None:
            df_main_effects.to_csv(df_main_effects)

        return df_main_effects
