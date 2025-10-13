# Python Implementation of the Rule–Stoffregen–Ermentrout Model of Stroboscopic Hallucinations
*A project for systematically exploring and generating simulations of flicker-induced geometric hallucinations.*

The original paper:
> Rule, M., Stoffregen, M., & Ermentrout, B. (2011). *A Model for the Origin and Properties of Flicker-Induced Geometric Phosphenes*. **PLoS Computational Biology, 7**(9), e1002158. https://doi.org/10.1371/journal.pcbi.1002158

## ⚠️ Safety
**Flashing-light sensitivity warning.** The GIFs contain high-frequency flashing images that may be harmful to photosensitive individuals. View with care.

## Installation

    git clone <https://github.com/ejgrove/rse_model.git>
    cd <rse_model>
    conda env create -f environment.yml
    conda activate rse-model

## Quick start
Run the simulation script in the CLI, or test out the notebook ```demo.ipynb```. 

CLI examples
```
# Generate cortical and retinal images
# --interval [INTERVAL]: sets time intervals (ms) of image sampling
# --end [END]: sets the duration (ms) of the simulation
# --images [IMAGES]: specifies whether to output the retinal, cortical, or both images
# --grid-size [GRID-SIZE]: specifies the sidelength of the neural field
```
```
python -m src.cli --interval 8000 --end 8000 --images both --label --grid-size 200
```
<img src="assets/images/cortical_8000ms.png" alt="Cortical Plot" width="350"> <img src="assets/images/retinal_8000ms.png" alt="Retinal Plot" width="350">


```
# Generate a plot
# --plot: Outputs plot
# --seed [SEED]: Sets random seed (default is none)
# --cmap [CMAP]: Sets matplotlib compatible colormap
# --T [T]: Sets the period (ms) of the stroboscopic stimulation
```
```
python -m src.cli --interval 8000 --end 8000 --plot --label --seed 42 --cmap nipy_spectral --T 55
```
<img src="assets/plots/plot_8000ms.png" alt="" width="700">


```
# Generate a GIF
```
```
python -m src.cli --end 8000 --gif --grid-size 100 --seed 43 --cmap nipy_spectral --T 50
```
[```assets/gifs/example4_progression_T50_nipy_spectral.gif```](assets/gifs/example4_progression_T50_nipy_spectral.gif) – WARNING: flashing content


- Try period (T) ranges 50–60 to produce roll planiforms and 110–130 for spots
- Adjust the size of the neural field (N) to see how it effects the formation of the patterns
- See [```cli.py```](src/cli.py) for description of all parameters

## More Examples

See [```assets/gifs```](assets/gifs) for more examples – WARNING: flashing content

## License
MIT. See [```LICENSE```](LICENSE/)