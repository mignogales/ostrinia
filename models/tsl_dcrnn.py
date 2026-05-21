from tsl.nn.models.stgn.dcrnn_model import DCRNNModel


class tsl_dcrnn(DCRNNModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)