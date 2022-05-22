"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from napari_plugin_engine import napari_hook_implementation
from napari import Viewer
from magicgui import magicgui
from typing import Any

simulation_devices = [('GPU', 'gpu'), ('CPU', 'cpu')]

def widget_wrapper():

    def generate_frames_and_background(device,
        number_of_emitters=10,
        lifetime_avg=1.0,
        height_size=32,
        width_size=32,
        number_of_frames=10,
        psf_extent_min_x=0,
        psf_extent_max_x=32,
        psf_extent_min_y=0,
        psf_extent_max_y=32,
        psf_extent_min_z=0,
        psf_extent_max_z=0,
        emitter_extent_min_x=0,
        emitter_extent_max_x=32,
        emitter_extent_min_y=0,
        emitter_extent_max_y=32,
        emitter_extent_min_z=0,
        emitter_extent_max_z=32,
        ):
            from napari.utils.notifications import show_info
            from napari import Viewer
            from decode.simulation import psf_kernel, structure_prior, emitter_generator, simulator, camera, background
            from munch import DefaultMunch
            import numpy as np
            
            dictionary = {
                    "Camera": {
                        "baseline": 398.6,
                        "convert2photons": True,
                        "e_per_adu": 5.0,
                        "em_gain": 100.0,
                        "px_size": [127.0, 117.0],
                        "qe": 1.0,
                        "read_sigma": 58.8,
                        "spur_noise": 0.0015,
                    },
                    "CameraPreset": None,
                    "Hardware": {
                        "device": device,
                        "device_simulation": device,
                        "num_worker_train": 2,
                        "torch_multiprocessing_sharing_strategy": None,
                        "torch_threads": 2,
                        "unix_niceness": 0,
                    },
                    "Meta": {"version": "0.10.0"},
                    "Simulation": {
                        "bg_uniform": [0, 100],
                        "density": None,
                        "emitter_av": number_of_emitters,
                        "emitter_extent": [[emitter_extent_min_x, emitter_extent_max_x], [emitter_extent_min_y, emitter_extent_max_y], [emitter_extent_min_z, emitter_extent_max_z]],
                        "img_size": [height_size, width_size],
                        "intensity_mu_sig": [7000.0, 3000.0],
                        "intensity_th": None,
                        "lifetime_avg": lifetime_avg,
                        "mode": "acquisition",
                        "photon_range": None,
                        "psf_extent": [[psf_extent_min_x, psf_extent_max_x], [psf_extent_min_y, psf_extent_max_y], [psf_extent_min_z, psf_extent_max_z]],
                        "roi_auto_center": False,
                        "roi_size": None,
                        "xy_unit": "px",
                    },
                    "TestSet": {
                        "frame_extent": [[-0.5, 39.5], [-0.5, 39.5], None],
                        "img_size": [40, 40],
                        "mode": "simulated",
                        "test_size": 512,
                    },
                }

            param = DefaultMunch.fromDict(dictionary)
            psf = psf_kernel.GaussianPSF(param.Simulation.psf_extent[0], param.Simulation.psf_extent[1], param.Simulation.psf_extent[2], (height_size, width_size), sigma_0=1.0)
            prior_struct = structure_prior.RandomStructure.parse(param)
            prior_train = emitter_generator.EmitterSamplerBlinking.parse(param, structure=prior_struct, frames=(0, number_of_frames))
            bg = background.UniformBackground.parse(param)

            if param.CameraPreset == 'Perfect':
                noise = camera.PerfectCamera.parse(param)
            elif param.CameraPreset is not None:
                raise NotImplementedError
            else:
                noise = camera.Photon2Camera.parse(param)

            simulation = simulator.Simulation(psf=psf, em_sampler=prior_train, background=bg, noise=noise, frame_range=(0, number_of_frames))

            emitter, frames, bg = simulation.sample()

            frames = frames.detach().cpu().numpy()
            bg = bg.detach().cpu().numpy()
            emitter = emitter.xyz_px.detach().cpu().numpy()

            return frames, bg, emitter

    @magicgui(
        call_button='Run DECODE Simulation',  
        layout='vertical',
        simulation_device = dict(widget_type='ComboBox', label='Device to run simulation', choices=simulation_devices, value='cpu', tooltip='Choose device to run simulation'),
        compute_diameter_button = dict(widget_type='PushButton', text='Generate datasets', tooltip='Giving the configuration, generate datasets')
    )

    def widget( 
        viewer: Viewer,
        simulation_device,
        compute_diameter_button,
        number_of_datasets_to_generate = 10,
        number_of_emitters=10,
        lifetime_avg=1.0,
        height_size=32,
        width_size=32,
        number_of_frames=10,
        psf_extent_min_x=0,
        psf_extent_max_x=32,
        psf_extent_min_y=0,
        psf_extent_max_y=32,
        psf_extent_min_z=0,
        psf_extent_max_z=0,
        emitter_extent_min_x=0,
        emitter_extent_max_x=32,
        emitter_extent_min_y=0,
        emitter_extent_max_y=32,
        emitter_extent_min_z=0,
        emitter_extent_max_z=32) -> None:

        frames, bg, _ = generate_frames_and_background(        
            simulation_device,
            number_of_emitters,
            lifetime_avg,
            height_size,
            width_size,
            number_of_frames,
            psf_extent_min_x,
            psf_extent_max_x,
            psf_extent_min_y,
            psf_extent_max_y,
            psf_extent_min_z,
            psf_extent_max_z,
            emitter_extent_min_x,
            emitter_extent_max_x,
            emitter_extent_min_y,
            emitter_extent_max_y,
            emitter_extent_min_z,
            emitter_extent_max_z
        )

        viewer.add_image(frames)

    @widget.compute_diameter_button.changed.connect 
    def generate_datasets(e: Any):
        from tifffile import imsave
        for i in range(widget.number_of_datasets_to_generate.value):
            frames, bg, _ = generate_frames_and_background('cpu')
            
            imsave('dataset_generated/' + str(i) + '_frames.tif', frames)
            imsave('dataset_generated/' + str(i) + '_backgrounds.tif', bg)            

    return widget            

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'decode simulator'}
