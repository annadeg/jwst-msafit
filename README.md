A forward modelling and fitting tool for NIRSpec MOS data. With msafit you can:

* Estimate the slit losses and line spread function for an arbitrary morphology in an arbitrary slit, for any wavelength and filter/disperser combination.

* Fit the spatial extent of one or more emission lines.

* Fit the 2D velocity field of one or more emission lines.

Documentation in progress... More examples will be uploaded in the next days/weeks.

Installation
------------

Dependencies: numpy, scipy, astropy. For fitting in 1D or 2D with lmfit, emcee, or ultranest, these packages need to be installed separately.

Download reference files from the following locations. Some of these files are large and not everything is needed: check which filter/disperser you are interested in and download only those traces and PSFs!

Files are hosted on Zenodo, which will become public once the arxiv ID is known.

Although you can specify the location of these files yourself when running the code, the following is much more convenient in the long run. Create a new folder (e.g. called "msafit_ref_data") that has two subfolders storing the detector properties (everything from Zenodo link 1) and PSF libraries (downloaded from Zenodo link 2):

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
export msa_refdata=/path/to/your/reference/data/
```

Install by downloading or cloning the repository (currently: the develop branch)

Then cd to msafit and:
```
python -m pip install .
```

You can try if the installation is successful by trying
```python
import msafit
```








