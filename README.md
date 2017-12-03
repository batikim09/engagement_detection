# engagement_detection
automatic engagement detection in a group of children

This is a repository for a joint paper between University of Twente and University of Bonn.

# Installation
Several python packages will be installed by the following command in a console:

sudo pip install -r requirements.txt

# Running sanity checks
open ./scripts/sanity_model_checks.sh

There are multiple experiments to check sanity.

First, check the feature DB path that may be different from the original setup.

In the script, replace the following option by your own path:

"../../features/ENGAGEMENT_DEEP_POSE/DEEPCP.POSE.func.indiv.nan.delta.h5"


# References

Bibtex:

@inproceedings{kim2017camera,
  title={Automatic ranking of engagement of a group of children ``in the wild'' using emotional states and deep pose features},
  author={Kim, Jaebok and H. Shareef, Ibrahim and Regier, Peter and P. Truong, Khiet and Charisi, Vicky and Zaga, Cristina and Bennewitz, Maren and Englebienne, Gwenn and Evers, Vanessa},
  booktitle={Proceedings of the workshop of CAMERA},
  pages={},
  year={2017}
}
