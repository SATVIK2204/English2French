English2French
==============================

## English to French translator using Sequence-to-Sequence Model with Attention.
It is a project to perform machine translation(English to french) using Seq2Seq model with Attention. The project uses modularized implementation of the same using pytorch.


## Setup
Make sure you have the required python dependencies for running the project.
Run `pip3 install -r requirements.txt` in your working environment to install the required pacakages.

I have not included pytorch package in the requirements.txt, so please install the pytorch version according to your cuda and cudnn versions. It is recommeneded to run the project on GPU(if available) for lower run time.

## How to produce the same result?
- First of all remove the data present in `models/interim` folder.
- Now open the notebook name `1.english2french.ipynb` present in notebooks folder.
- Now run the cells and follow the comments in it.
- You can do the respective imported module path to see its implementatiomn.

## Future Works
Working on the deployment of the project to take user's input and ouput the respective translation.


## Contributing
As this project is still under, it would be much appericated if you submit a PR for cleanups, error-fixing, or adding new (relevant) content.


--------


