import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DNN(pl.LightningModule):

    def __init__(
        self,
        c_in: int,
        c_hidden: int,
        c_out: int,
        num_layers: int = 4,
        dropout: float = 0.3
    ) -> None:
        """Creates a diabetes neural network based on the parameters.

        Args:
            c_in (int): Inner channels used in the first layer.
            c_hidden (int): Hidden channels used to build the hidden layers based
                on the number of layers passed. 
            c_out (int): Out channels used to build the out layer of the model,
                in case of classification layer provide the number of classes
                that you have if $class\_number > 2$ 
            num_layers (int, optional): Number of hidden layers used to build
                the model. Defaults to 4.
            dropout (float, optional): Dropout rate used in the output head of
                the model. Defaults to 0.3.
        """
        super().__init__()
        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        
        self.loss_module = (
            nn.BCEWithLogitsLoss() if c_out == 1
            else nn.CrossEntropyLoss()
        )

        layers = []
        channels_in, channels_out = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_features=channels_in, out_features=channels_out),
                nn.ReLU(),
            ]
            channels_in = channels_out

        layers += [
            nn.Linear(in_features=channels_in, out_features=channels_out)
        ]

        self.layers = nn.ModuleList(layers)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=c_hidden, out_features=c_out)
        )

    def forward(self, x, mode: str = 'train'):
        x = self.flatten(x)

        for layer in self.layers:
            x = layer(x)

        x = self.head(x)
        return x.squeeze(dim=-1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        x = self.forward(data, mode='train')

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            target = target.squeeze(dim=-1).float()
        else:
            preds = x.argmax(dim=-1)

        loss = self.loss_module(x, target)
        acc = (preds == target).sum().float() / preds.shape[0]

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        x = self.forward(data, mode='val')

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            target = target.squeeze(dim=-1).float()
        else:
            preds = x.argmax(dim=-1)
        
        acc = (preds == target).sum().float() / preds.shape[0]
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        data, target = batch
        x = self.forward(data, mode='test')

        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            target = target.squeeze(dim=-1).float()
        else:
            preds = x.argmax(dim=-1)

        acc = (preds == target).sum().float() / preds.shape[0]
        self.log('test_acc', acc)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.0)
