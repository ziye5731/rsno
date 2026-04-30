from prepare_data import *
import torch
import lightning as L
import os
import gc

def set_exp_dir(exp_base_dir, exp_name):
    '''
    Set the experiment directory based on the base directory and experiment name.
    If the directory already exists, it appends a number to create a unique directory.
    '''
    os.makedirs(exp_base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(exp_base_dir) if d.startswith(exp_name) and d[len(exp_name):].isdigit()]
    n = max([int(d[len(exp_name):]) for d in existing_dirs], default=0) + 1
    exp_dir = os.path.join(exp_base_dir, f"{exp_name}{n}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_results(results, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i, result in enumerate(results):
        torch.save(result, os.path.join(save_dir, f'result_{i}.pt'))

    print(f"Results saved to {save_dir}")


exp = 'discrete_ssr'  # 'discrete_ssr', 'continuous_ssr_2x', 'continuous_ssr_4x'.

rsno_dir = 'path/to/.ckpt'  # the path to the trained RSNO model checkpoint.


# ----------------- Set up experiment directory -----------------
exp_base_dir = f'./inference_results'
exp_name = f'{exp}'
exp_dir = set_exp_dir(exp_base_dir, exp_name)
# write model dir to the exp_dir
with open(os.path.join(exp_dir, 'model_dirs.txt'), 'w') as f:
    f.write(f'rsno: {rsno_dir}\n')


# ----------------- Load models -----------------
def load_rsno(ckpt_path):
    from rsno.rsno import RSNO
    from rsno.neuralops import CatUNO
    from rsno_lightning import RSNOLightningModule

    import pandas as pd

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

    rsno_uno = RSNO(
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
        refinement=True
    )

    rsno_lightning = RSNOLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=rsno_uno,
    )

    return rsno_lightning

if os.path.isfile(rsno_dir):
    rsno_lightning = load_rsno(rsno_dir)



# ----------------- Load datasets -----------------
train_dataset, val_dataset, test_dataset, \
train_dataloader, val_dataloader, test_dataloader, \
val_dataloaders, test_dataloaders = get_data(exp, batch_size=1)



# ----------------- Predict on test dataset, and save results -----------------
trainer = L.Trainer(
    accelerator='gpu',
    devices='auto',
)

trainer_single_gpu = L.Trainer(
    accelerator='gpu',
    devices=1,
)


for idx, test_dataloader in enumerate(test_dataloaders):
    rsno_result = None
    if os.path.isfile(rsno_dir):
        rsno_result = trainer_single_gpu.predict(
            model=rsno_lightning,
            dataloaders=test_dataloader,
        )
        if rsno_result is not None:
            save_results(rsno_result, os.path.join(exp_dir, f'rsno/site_{idx}'))         
    del rsno_result
    torch.cuda.empty_cache()
    gc.collect()

print('Finished.')
