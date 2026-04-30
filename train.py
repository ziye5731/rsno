import os
import sys
# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsno.rsno import RSNO
from rsno.neuralops import CatUNO
from rsno_lightning import RSNOLightningModule
import pandas as pd
import lightning as L
from lightning.pytorch import loggers
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


ckpt_path = None
exp = 'discrete_ssr'  # 'discrete_ssr', 'continuous_ssr_2x', 'continuous_ssr_4x'

batch_size = 2

# Determine the experiment directory
os.makedirs('./exp_results', exist_ok=True)
exp_base_dir = f'./exp_results/{exp}'
existing_dirs = [d for d in os.listdir('./exp_results') if d.startswith(f'{exp}') and d[len(f'{exp}'):].isdigit()]
n = max([int(d[len(f'{exp}'):]) for d in existing_dirs], default=0) + 1
exp_dir = f"{exp_base_dir}{n}"
os.makedirs(exp_dir, exist_ok=True)

early_stop = EarlyStopping(monitor="val0_PSNR", patience=10, mode="max")
check_point = ModelCheckpoint(
    dirpath=exp_dir,
    filename='epoch={epoch:02d}-step={step}',
    monitor="val0_PSNR",
    mode="max",
    save_top_k=1,
    save_last=True,
)

max_epochs = 200
trainer = L.Trainer(
    accelerator='gpu',
    devices='auto',
    max_epochs=max_epochs,
    logger=loggers.CSVLogger(
        save_dir=exp_dir,
    ),
    log_every_n_steps=100,
    precision='32',
    accumulate_grad_batches=8//batch_size,
    callbacks=[early_stop, check_point],
)

learning_rate = 5e-4  # Learning rate for the optimizer


# --------------------- Preparing Datasets ---------------------
from prepare_data import *

train_dataset, val_dataset, test_dataset, \
train_dataloader, val_dataloader, test_dataloader, \
val_dataloaders, test_dataloaders = get_data(exp, batch_size=batch_size)


# --------------------- Load Model ---------------------

# Load the SMARTS2 data from smarts295.ext.txt
data = pd.read_csv('./rsno/SMARTS2/smarts295.ext.txt', sep=' ')
data.columns = [
    'Wvlgth', 
    'Direct_normal_irradiance',
    'Difuse_tilted_irradiance', 
    'Global_tilted_irradiance',
    'Beam_normal_+circumsolar',
    'Difuse_horiz-circumsolar',
    'Zonal_ground_reflectance'
]
data['Wvlgth'] = data['Wvlgth'] * 0.001  # Convert to micrometers
smarts_data = data

prior_data = pd.DataFrame({
    'wavelength': smarts_data['Wvlgth'],
    'value': smarts_data['Direct_normal_irradiance']
})


width = 32
modes = 16

rsno = RSNO(
    prior_data=prior_data,
    no=CatUNO(
        d_r=1,
        d_s=1,
        d_out=1,
        width=width,
        layers_c=4,
        layers_t=4,
        modes=modes
    ),
)

steps_per_epoch = len(train_dataloader)

total_steps = steps_per_epoch * max_epochs

rsno_lightning = RSNOLightningModule(
    model=rsno,
    lr=learning_rate,
    max_epochs=max_epochs,
    steps_per_epoch=steps_per_epoch
)



# --------------------- Training ---------------------
print(f"Train dataset size: {len(train_dataset)}")

trainer.fit(
    model=rsno_lightning,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    ckpt_path=ckpt_path
)


# --------------------- Testing ---------------------
trainer.test(
    model=rsno_lightning,
    dataloaders=test_dataloader,
)

trainer.test(
    model=rsno_lightning,
    dataloaders=test_dataloaders,
)
