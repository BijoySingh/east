Emotion And Sentiment Toolkit
=============================
### AUTHORS
* Bijoy Singh Kochar
* Prof. Pushpak Bhattacharya *(Guide)*
* Aditya Joshi *(Co-Guide)*

### OTHER CONTRIBUTORS
* Dheerendra Rathor

Created as a part of the B.Tech. Project at IITBombay, with Prof. Pushpak Bhattacharya

### SETUP PROCEDURE

```
git clone https://github.com/BijoySingh/east.git
cd east
pip install . [or] python setup.py install
```

Requirements

```
nltk
numpy
scipy
scikit-learn
progressbar
```

### USAGE
Command Line
```
$ east --help
$ east -i="Hello this is a sentence to test"
```

Python Code
```
from east.toolkit import Toolkit
toolkit = Toolkit()
toolkit.analyse("This is some document string........")
```