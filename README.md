# Python Implementation of the Rule–Stoffregen–Ermentrout (2011) Neural Field Model of Stroboscopic Hallucinations
*Designed for systematic exploration of stroboscopically-induced geometric hallucinations under controlled parameter regimes.*

The original paper:
> Rule, M., Stoffregen, M., & Ermentrout, B. (2011). *A Model for the Origin and Properties of Flicker-Induced Geometric Phosphenes*. **PLoS Computational Biology, 7**(9), e1002158. https://doi.org/10.1371/journal.pcbi.1002158

## ⚠️ Safety
**Flashing-light sensitivity warning.** The GIFs contain high-frequency flashing images that may be harmful to photosensitive individuals. View with care.

## Installation
The code is structured as a reusable simulation package with a command-line interface for reproducible experiments.

    git clone https://github.com/ejgrove/rse_model.git
    cd rse_model
    conda env create -f environment.yml
    conda activate rse-model

## Quick start
The command-line interface [`(cli.py)`](src/cli.py) supports robust and reproducible simulations. A demonstration notebook [`(demo.ipynb)`](notebooks/demo.ipynb) is provided for interactive exploration.

**See [```cli.py```](src/cli.py) for description of all parameters**

### CLI examples

### Cortical and Retinal images
```
python -m src.cli --interval 8000 --end 8000 --images both --label --N 201
```

<img src="assets/images/cortical_8000ms.png" alt="Cortical Plot" width="350"> <img src="assets/images/retinal_8000ms.png" alt="Retinal Plot" width="350">

*The cortical plot shows activity in visual cortical coordinates, while the retinal plot applies the inverse retino-cortical transform to approximate the perceived hallucination.*


### Plots
```
python -m src.cli --interval 8000 --end 8000 --plot --label --seed 42 --cmap nipy_spectral --T 55
```

<img src="assets/plots/plot_8000ms.png" alt="" width="700">


### GIFs
```
python -m src.cli --end 8000 --gif --N 101 --seed 43 --cmap nipy_spectral --T 50
```

[```assets/gifs/example4_progression_T50_nipy_spectral.gif```](assets/gifs/example4_progression_T50_nipy_spectral.gif) – WARNING: flashing content


## Tips
- Periods (`--T`) in the range 50–60 ms tend to produce roll-like planforms, while periods around 110–130 ms often yield hexagonal patterns.
- Adjust the size of the neural field (`--N`) to increase the spatial frequency of the patterns. However, increasing `--N` above 250 reduces the stability of the pattern formation.

## More Examples
See [`assets/gifs`](assets/gifs) for more examples – WARNING: flashing content

## License
MIT. See [`LICENSE`](LICENSE/)