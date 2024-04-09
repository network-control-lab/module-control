# Module control of complex network

#### Calculate the control powers between modules (obtained by Louvain algorithm) based on the minimum dominating set (MDSet).

### This method was cited in:
#### 1. **_"Module control of network analysis in psychopathology." Pan et al._**

### Project explanation:
#### 1. "/main.py"
/main.py stores the main functions of the algorithm, including building symptom networks, calculating modules based on the Louvain algorithm, calculating control powers between modules, etc.
#### 2. "/robust_test.py"
/robust_test.py is mainly used to analyze the stability of module control characteristics. Based on Bootstrapping, it analyzes network edge stability (weight and number), MDSets stability (size and number), module stability (size and number), etc.
