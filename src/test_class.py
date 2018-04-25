import pytest
import prediction_model


class TestClass:

    pd = prediction_model.prediction_model()

    def test_candle_data(self):
        assert len(self.pd.data.candle_data[0]) == 3

    def test_norm_data(self):
        assert len(
            (self.pd.data.norm_data_ls[0]['High'].values > 1.01).nonzero()[0]) == 0
        assert len(
            (self.pd.data.norm_data_ls[0]['Low'].values > 1.01).nonzero()[0]) == 0
        assert len(
            (self.pd.data.norm_data_ls[0]['Open'].values > 1.01).nonzero()[0]) == 0
        assert len(
            (self.pd.data.norm_data_ls[0]['Close'].values > 1.01).nonzero()[0]) == 0

    def test_valid_split(self):
        assert self.pd.data.split > 0

    def test_get_data(self):
        assert len(self.pd.data.data_ls) > 0

    def test_in_size(self):
        assert self.pd.data.input_size > 1

    def test_learning_rate(self):
        assert self.pd.optimizer.param_groups[0]['lr'] > 0
