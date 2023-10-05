from setuptools import setup, find_packages

with open("Readme.md", encoding='utf8') as fh:
    long_description = fh.read()

setup(
    name="hvsrUNet",
    version="0.0.1",
    author="Zhiyu Zhang",
    author_email="erbiaoger@gmail.com",
    url="https://github.com/erbiaoger/hvsrUNet",
    description="hvsrUNet - open source software for HVSR analysis with deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    #packages=['hvsrUNet'],
    packages=find_packages(),
    package_data={'hvsrUNet': [
                            ]},
    data_files=[
    ],

    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',

        "Operating System :: OS Independent",
    ],
    project_urls={
        'Bug Reports': 'https://github.com/erbiaoger/hvsrUNet/issues',
        'Source': 'https://github.com/erbiaoger/hvsrUNet',
        'Docs': 'https://github.com/erbiaoger/hvsrUNet/docs',
    },

    keywords='horizontal-to-vertical spectral ratio hv hvsr deep learning',
    #install_requires=['torch', 'matplotlib', 'sklearn'],
    #entry_points={'console_scripts': ['hvsrUNet = hvsrUNet.__main__:main']},
    # entry_points={
    #     'console_scripts': [
    #         'hvsrUNet = hvsrUNet.cli:cli'
    #     ],
    # },
)
