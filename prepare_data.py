import datasets
import os
import torch
import torch.utils.data


img_shape = (64, 64)

data_dir = 'path/to/AVIRIS'  # the directory where the AVIRIS data is stored. It should contain 6 subdirectories: Madison, Blodgett_Forest, Cuprite, Everglades, Yellowstone, and Los_Angeles. 
css_dir = 'path/to/camSpecSensitivity'  # the directory where the camera spectral sensitivity data is stored. 

aviris_madison_datasets = {}
for scene in os.listdir(f'{data_dir}/Madison'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Madison', scene)):
        continue
    aviris_madison_datasets[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Madison', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Madison/wavelength.pt'
    )

aviris_blodgett_forest_dataset = {}
for scene in os.listdir(f'{data_dir}/Blodgett_Forest'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Blodgett_Forest', scene)):
        continue
    aviris_blodgett_forest_dataset[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Blodgett_Forest', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Blodgett_Forest/wavelength.pt'
    )

aviris_cuprite_dataset = {}
for scene in os.listdir(f'{data_dir}/Cuprite'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Cuprite', scene)):
        continue
    aviris_cuprite_dataset[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Cuprite', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Cuprite/wavelength.pt'
    )

aviris_everglades_dataset = {}
for scene in os.listdir(f'{data_dir}/Everglades'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Everglades', scene)):
        continue
    aviris_everglades_dataset[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Everglades', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Everglades/wavelength.pt'
    )

aviris_yellowstone_dataset = {}
for scene in os.listdir(f'{data_dir}/Yellowstone'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Yellowstone', scene)):
        continue
    aviris_yellowstone_dataset[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Yellowstone', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Yellowstone/wavelength.pt'
    )

aviris_los_angeles_dataset = {}
for scene in os.listdir(f'{data_dir}/Los_Angeles'):
    if not os.path.isdir(os.path.join(f'{data_dir}/Los_Angeles', scene)):
        continue
    aviris_los_angeles_dataset[scene] = datasets.aviris.AVIRIS_scenes_SSR(
        data_dir=os.path.join(f'{data_dir}/Los_Angeles', scene),
        srf_dir=f'{css_dir}',
        crop_size=img_shape[0],
        wvlgth_dir=f'{data_dir}/Los_Angeles/wavelength.pt'
    )



def make_inter_datasets(scale=2):
    aviris_madison_datasets_inter = {}
    for scene in os.listdir(f'{data_dir}/Madison'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Madison', scene)):
            continue
        aviris_madison_datasets_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Madison', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Madison/wavelength.pt',
            scale=scale
        )

    aviris_blodgett_forest_dataset_inter = {}
    for scene in os.listdir(f'{data_dir}/Blodgett_Forest'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Blodgett_Forest', scene)):
            continue
        aviris_blodgett_forest_dataset_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Blodgett_Forest', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Blodgett_Forest/wavelength.pt',
            scale=scale
        )

    aviris_cuprite_dataset_inter = {}
    for scene in os.listdir(f'{data_dir}/Cuprite'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Cuprite', scene)):
            continue
        aviris_cuprite_dataset_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Cuprite', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Cuprite/wavelength.pt',
            scale=scale
        )

    aviris_everglades_dataset_inter = {}
    for scene in os.listdir(f'{data_dir}/Everglades'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Everglades', scene)):
            continue
        aviris_everglades_dataset_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Everglades', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Everglades/wavelength.pt',
            scale=scale
        )

    aviris_yellowstone_dataset_inter = {}
    for scene in os.listdir(f'{data_dir}/Yellowstone'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Yellowstone', scene)):
            continue
        aviris_yellowstone_dataset_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Yellowstone', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Yellowstone/wavelength.pt',
            scale=scale
        )

    aviris_los_angeles_dataset_inter = {}
    for scene in os.listdir(f'{data_dir}/Los_Angeles'):
        if not os.path.isdir(os.path.join(f'{data_dir}/Los_Angeles', scene)):
            continue
        aviris_los_angeles_dataset_inter[scene] = datasets.aviris.AVIRIS_scenes_Interpolation(
            data_dir=os.path.join(f'{data_dir}/Los_Angeles', scene),
            srf_dir=f'{css_dir}',
            crop_size=img_shape[0],
            wvlgth_dir=f'{data_dir}/Los_Angeles/wavelength.pt',
            scale=scale
        )

    return aviris_madison_datasets_inter, aviris_blodgett_forest_dataset_inter, \
           aviris_cuprite_dataset_inter, \
           aviris_everglades_dataset_inter, aviris_yellowstone_dataset_inter, \
           aviris_los_angeles_dataset_inter



def get_data(exp, batch_size=1):
    if exp == 'discrete_ssr':
        train_dataset = torch.utils.data.ConcatDataset([
            aviris_madison_datasets['f110816t01p00r06rdn_c'],
            aviris_madison_datasets['f110816t01p00r08rdn_c'],
            aviris_blodgett_forest_dataset['f130803t01p00r02rdn_e'],
            aviris_cuprite_dataset['f060502t01p00r04rdn_c'],
            aviris_cuprite_dataset['f060502t01p00r05rdn_c'],
            aviris_cuprite_dataset['f060502t01p00r06rdn_c'],
            aviris_everglades_dataset['f100523t01p00r13rdn_b'],
            aviris_everglades_dataset['f100523t01p00r15rdn_b'],
            aviris_yellowstone_dataset['f060925t01p00r06rdn_c'],
            aviris_los_angeles_dataset['f130821t01p00r04rdn_e'],
        ])
        val_dataset = [
            aviris_madison_datasets['f110816t01p00r09rdn_c'],
            aviris_madison_datasets['f110816t01p00r10rdn_c'],
            aviris_blodgett_forest_dataset['f130803t01p00r03rdn_e'],
            aviris_cuprite_dataset['f060502t01p00r07rdn_c'],
            aviris_everglades_dataset['f100523t01p00r12rdn_b'],
            aviris_yellowstone_dataset['f060925t01p00r08rdn_c'],
            aviris_los_angeles_dataset['f130821t01p00r06rdn_e'],
        ]
        test_dataset = [
            torch.utils.data.ConcatDataset([
                aviris_madison_datasets['f110816t01p00r11rdn_c'],
                aviris_madison_datasets['f110816t01p00r12rdn_c'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_blodgett_forest_dataset['f130803t01p00r04rdn_e']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_cuprite_dataset['f060502t01p00r08rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_everglades_dataset['f100523t01p00r14rdn_b'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_yellowstone_dataset['f060925t01p00r10rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_los_angeles_dataset['f130821t01p00r08rdn_e']
            ]),
        ]

    elif exp == 'continuous_ssr_2x':
        aviris_madison_datasets_inter, aviris_blodgett_forest_dataset_inter, \
        aviris_cuprite_dataset_inter, \
        aviris_everglades_dataset_inter, aviris_yellowstone_dataset_inter, \
        aviris_los_angeles_dataset_inter = make_inter_datasets(scale=2)

        train_dataset = torch.utils.data.ConcatDataset([
            aviris_madison_datasets_inter['f110816t01p00r06rdn_c'],
            aviris_madison_datasets_inter['f110816t01p00r08rdn_c'],
            aviris_blodgett_forest_dataset_inter['f130803t01p00r02rdn_e'],
            aviris_cuprite_dataset_inter['f060502t01p00r04rdn_c'],
            aviris_cuprite_dataset_inter['f060502t01p00r05rdn_c'],
            aviris_cuprite_dataset_inter['f060502t01p00r06rdn_c'],
            aviris_everglades_dataset_inter['f100523t01p00r13rdn_b'],
            aviris_everglades_dataset_inter['f100523t01p00r15rdn_b'],
            aviris_yellowstone_dataset_inter['f060925t01p00r06rdn_c'],
            aviris_los_angeles_dataset_inter['f130821t01p00r04rdn_e'],
        ])
        val_dataset = [
            aviris_madison_datasets_inter['f110816t01p00r09rdn_c'],
            aviris_madison_datasets_inter['f110816t01p00r10rdn_c'],
            aviris_blodgett_forest_dataset_inter['f130803t01p00r03rdn_e'],
            aviris_cuprite_dataset_inter['f060502t01p00r07rdn_c'],
            aviris_everglades_dataset_inter['f100523t01p00r12rdn_b'],
            aviris_yellowstone_dataset_inter['f060925t01p00r08rdn_c'],
            aviris_los_angeles_dataset_inter['f130821t01p00r06rdn_e'],
        ]
        test_dataset = [
            torch.utils.data.ConcatDataset([
                aviris_madison_datasets['f110816t01p00r11rdn_c'],
                aviris_madison_datasets['f110816t01p00r12rdn_c'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_blodgett_forest_dataset['f130803t01p00r04rdn_e']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_cuprite_dataset['f060502t01p00r08rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_everglades_dataset['f100523t01p00r14rdn_b'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_yellowstone_dataset['f060925t01p00r10rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_los_angeles_dataset['f130821t01p00r08rdn_e']
            ]),
        ]
    
    elif exp == 'continuous_ssr_4x':
        aviris_madison_datasets_inter, aviris_blodgett_forest_dataset_inter, \
        aviris_cuprite_dataset_inter, \
        aviris_everglades_dataset_inter, aviris_yellowstone_dataset_inter, \
        aviris_los_angeles_dataset_inter = make_inter_datasets(scale=4)

        train_dataset = torch.utils.data.ConcatDataset([
            aviris_madison_datasets_inter['f110816t01p00r06rdn_c'],
            aviris_madison_datasets_inter['f110816t01p00r08rdn_c'],
            aviris_blodgett_forest_dataset_inter['f130803t01p00r02rdn_e'],
            aviris_cuprite_dataset_inter['f060502t01p00r04rdn_c'],
            aviris_cuprite_dataset_inter['f060502t01p00r05rdn_c'],
            aviris_cuprite_dataset_inter['f060502t01p00r06rdn_c'],
            aviris_everglades_dataset_inter['f100523t01p00r13rdn_b'],
            aviris_everglades_dataset_inter['f100523t01p00r15rdn_b'],
            aviris_yellowstone_dataset_inter['f060925t01p00r06rdn_c'],
            aviris_los_angeles_dataset_inter['f130821t01p00r04rdn_e'],
        ])
        val_dataset = [
            aviris_madison_datasets_inter['f110816t01p00r09rdn_c'],
            aviris_madison_datasets_inter['f110816t01p00r10rdn_c'],
            aviris_blodgett_forest_dataset_inter['f130803t01p00r03rdn_e'],
            aviris_cuprite_dataset_inter['f060502t01p00r07rdn_c'],
            aviris_everglades_dataset_inter['f100523t01p00r12rdn_b'],
            aviris_yellowstone_dataset_inter['f060925t01p00r08rdn_c'],
            aviris_los_angeles_dataset_inter['f130821t01p00r06rdn_e'],
        ]
        test_dataset = [
            torch.utils.data.ConcatDataset([
                aviris_madison_datasets['f110816t01p00r11rdn_c'],
                aviris_madison_datasets['f110816t01p00r12rdn_c'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_blodgett_forest_dataset['f130803t01p00r04rdn_e']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_cuprite_dataset['f060502t01p00r08rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_everglades_dataset['f100523t01p00r14rdn_b'],
            ]),
            torch.utils.data.ConcatDataset([
                aviris_yellowstone_dataset['f060925t01p00r10rdn_c']
            ]),
            torch.utils.data.ConcatDataset([
                aviris_los_angeles_dataset['f130821t01p00r08rdn_e']
            ]),
        ]

    else:
        raise ValueError(f"Unknown experiment setting: {exp}.")


    for dataset in val_dataset:
        dataset.crop_size = None
    for dataset in test_dataset:
        if hasattr(dataset, 'crop_size'):
            dataset.crop_size = None
        elif type(dataset) is list:
            for scene in dataset:
                scene.crop_size = None
        elif type(dataset) is torch.utils.data.ConcatDataset:
            for scene in dataset.datasets:
                scene.crop_size = None


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()//2)

    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(val_dataset), 
        batch_size=batch_size, 
        num_workers=os.cpu_count()//2
    )

    val_dataloaders = [
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count()//2)
        for dataset in val_dataset
    ]

    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(test_dataset), 
        batch_size=batch_size, 
        num_workers=os.cpu_count()//2
    )

    test_dataloaders = [
        torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count()//2)
        for dataset in test_dataset
    ]

    return train_dataset, val_dataset, test_dataset, \
           train_dataloader, val_dataloader, test_dataloader, \
           val_dataloaders, test_dataloaders