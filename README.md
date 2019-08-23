# GammaBoard

_A dashboard to show them all._


GammaBoard is a simple jupyter dashboard thought to display metrics assessing the reconstructions performances of 
Imaging Atmospheric Cherenkov Telescopes (IACTs).   
Deep learning is a lot about bookkeeping and trials and errors. GammaBoard ease this bookkeeping and allows quick 
comparison of the reconstruction performances of your machine learning experiments.

It is a working prototype used in CTA, especially by the [GammaLearn](https://gitlab.lapp.in2p3.fr/GammaLearn/) project

## Dependencies
- Matplotlib
- Scikit-learn
- Pandas
- Pytables
- ctaplot (>=0.3.0)


## Install

```
pip install gammaboard
```

```
export GAMMABOARD_DATA=path_to_the_data_directory
```

We recommend that you add this line to your bash source file (`$HOME/.bashrc` or `$HOME/.bash_profile`)

## Demo

Here is a simple demo of GammaBoard.
- On top the plots (metrics) such as angular resolution and energy resolution.
- Below, the list of experiments in the user folder.

When an experiment is selected in the list, the data is automatically loaded, the metrics computed and displayed.
A list of information provided during the training phase is also displayed.    
As many experiments results can be overlaid.     
When an experiment is deselected, it simply is removed from the plots.



![gammaboard_demo](share/gammaboard.gif)
