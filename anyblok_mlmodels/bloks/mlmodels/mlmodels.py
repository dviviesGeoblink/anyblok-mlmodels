"""Prediction model
"""
from anyblok import Declarations
from anyblok.columns import String

from logging import getLogger

logger = getLogger(__name__)
Model = Declarations.Model


@Declarations.register(Model)
class PredictionModel:
    """PredictionModel
    """
    model_name = String(label='Model name', nullable=False)

    def __str__(self):
        return 'Model {}'.format(self.model_name)

    def __repr__(self):
        msg = ('<PredictionModel: {self.model_name}>')

        return msg.format(self=self)
