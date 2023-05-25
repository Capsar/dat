from setuptools import setup, find_packages

setup(
    name='dat-trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'torchvision',	
        'numpy',
        'tqdm',
        'requests',
        'wandb',
    ],
    # entry_points={
    #         'console_scripts': [
    #             'dat_package=trainer.task:main',
    #         ],
    # },
    description='Training code application for DAT (Distributed Adversarial Training). Based on paper of Distributed Adversarial Training to Robustify Deep Neural Networks at Scale.'    
)