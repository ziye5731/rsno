import lightning as L
import torch
import torch.nn.functional as F
import loss as l
import metrics


METRICS = [getattr(metrics, func) for func in dir(metrics) if callable(getattr(metrics, func))]



class RSNOLightningModule(L.LightningModule):
    def __init__(self, model, max_epochs=400, steps_per_epoch=1000, lr=1e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss = l.L1_SAM_Loss()

        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inp, label = batch
        output = self(inp)
        loss = self.loss(output, label[0])
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inp, y = batch
        output = self(inp)
        self.log(f'val{dataloader_idx}_loss', self.loss(output, y[0]))
        
        for metric in METRICS:
            self.log(f'val{dataloader_idx}_{metric.__name__}', metric(output, y[0]).mean())

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        inp, y = batch
        output = self(inp)
        self.log(f'test{dataloader_idx}_loss', self.loss(output, y[0]))
        
        for metric in METRICS:
            self.log(f'test{dataloader_idx}_{metric.__name__}', metric(output, y[0]).mean())
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inp, y = batch
        rgb, coord, srf = inp
        hsi, _ = y

        h, w = hsi.shape[-2], hsi.shape[-1]
        C = hsi.shape[-3]

        hsi_pred = self(inp)  # [B, C, H, W]
        rgb_pred = torch.matmul(srf, hsi_pred.contiguous().view(C, -1)).view(-1, h, w)  # [B, M, H, W]

        return {
            'hsi_pred': hsi_pred,
            'rgb_pred': rgb_pred,
            'hsi': hsi,
            'rgb': rgb,

            'coord': coord,
            'srf': srf,
            'model': self._get_name()
        }

    

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs * self.steps_per_epoch,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
