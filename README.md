msafit provides a forward modelling and fitting tool for JWST/NIRSpec MOS data. This software is described in de Graaff et al. (2023): https://arxiv.org/abs/2308.09742. If you use msafit for your research, please cite this paper.

With msafit you can:

* Estimate the slit losses and line spread function for a user-specified morphology in any slit, for any wavelength and filter/disperser combination.

* Fit the spatial extent of one or more emission lines.

* Fit the 2D velocity field of one or more emission lines.

Documentation in progress... One example script is currently available, with more examples to be uploaded in the near future - stay tuned!

Installation
------------

Dependencies: numpy, scipy, astropy. For fitting in 1D or 2D with lmfit, emcee, or ultranest, these packages need to be installed separately.

First, download reference files needed to run the software. These files are hosted on Zenodo. Some of the files are large and not everything is needed: check which filter/disperser you are interested in and download only those traces and PSFs if you want to save disk space!

* trace libraries and detector properties: https://doi.org/10.5281/zenodo.8265895

* model PSF libraries: https://doi.org/10.5281/zenodo.8265441


Although you can specify the location of these files yourself when running the code, the following is much more convenient in the long run. Create a new folder (e.g. called "msafit_ref_data") that has two subfolders storing the detector properties (everything from Zenodo link 1) and PSF libraries (downloaded from Zenodo link 2). The file structure should then look like this:

msafit_ref_data/ \
├── detector/ \
│   ├── coordinates.fits \
│   ├── kernel_sca491.txt \
│   ├── kernel_sca492.txt \
│   ├── properties.txt \
│   ├── trace_lib_CLEAR_PRISM.fits \
│   └── ... \
├── psf/ \
│   ├── 1x1_PRISM_Q3_PSFLib.fits \
│   ├── ... \

Then set an environment variable ($msa_refdata) in your bashrc (or similar) pointing to this directory:
```bash
export msa_refdata=/path/to/my/msafit_ref_data/
```

Install the software by downloading or cloning the repository. Then cd to jwst-msafit and type:
```
python -m pip install .
```

You should now be able to import the package by typing

```python
import msafit
```







