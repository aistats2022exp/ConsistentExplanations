# Consistent Sufficient Explantions and Minimal Sufficient Rules 
 
Active Coalition of Variables (ACV) is a Python Package that aims to explain any machine learning model or data. 
It implemented the SDP explanations approaches of the Paper: [ref].
 
## Requirements
Python 3.6+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

## Installation

Clone the repo and run the following command in the main directory
```
$ python setup.py install
```

## How does ACV work?
To compute the different explanations, we only need a **trained Random Forest** and 
- Data **(X, Y)** if we want to explain directly the data
- Or **(X, f(X))** if we want to explain the model f

### Example:
In the following examples, we assume that we want to explain the test set (x_test) given the training set (x_train).
```python
from acv_explainers import ACVTree

forest = RandomForestClassifier() # or  Random Forest Regressor models
#...trained the model

# Initialize the explainer
acvtree = ACVTree(forest, x_train) # data should be np.ndarray with dtype=double
```
The main tool of our explanations is the Same Decision Probability (SDP). Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we fixed variables 
<img src="https://latex.codecogs.com/gif.latex?X_S = x_{S}" /> or when we do not observe the variables <img src="https://latex.codecogs.com/gif.latex?X_{\bar{S}}" />.
* **How to compute  the Same Decision Probability of a subset S <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
sdp = acvtree.compute_sdp_rf(x_test, y_test, x_train, y_train, S)

"""
Description of the arguments    
S (np.ndarray[1]): index of variables on which we want to compute the SDP
"""
```
* **How to compute all the Sufficient Explanations <img src="https://latex.codecogs.com/gif.latex?S^\star" /> 
and the Local Explanatory Importance** of each instance ?
```python
# all the sufficient explanations 
sufficient_coal, sdp_coal, sdp_global = acvtree.sufficient_coal_rf(x_test, y_test, x_train, y_train,
                                             global_proba=global_proba, classifier=0, t=t)
"""
Description of the arguments
global_proba (double): the minimal level of the SDP, default value=0.9
classifier (int): 0 if it is a classification problem, else 1 if it is a regression problem
t (np.double): only necessary for regression problem, it corresponds to the radius of the ball of the SDP 

sufficient_coal (list(list)): the different sufficient explanations (by index) for each instance in x_test
"""

# Local explanatory importance
local_importance = acvtree.compute_local_importance(d, sufficient_coal)
```

*  **How to compute the Minimal Sufficient Explanation ?**
```python
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_rf(x_test, y_test, x_train, y_train, data, C=[[]], global_proba=0.9)

"""
Description of the arguments
classifier (int): 0 if it is a classification problem else 1 if it is a regression problem
S (list[list]): list containing the index of variables S for each observation
pi (double): the level of the SDP, default value = 0.9
t (double): the radious of the ball of the SDP 
C (list[list]): list of the index of variables group together

sdp_index[i, :size[i]] corresponds to the index of the variables in $S^\star$ of observation i  
sdp[i] corresponds to the SDP value of the $S^\star$ of observation i
"""
```

* **How to compute the Minimal Local Rule based on a Sufficient Explanation S**

```python
sdp, rules, sdp_all, rules_data = acvtree.compute_sdp_maxrules(x_test, y_test, x_train, y_train,
                                                    S=S, classifier=0, t=t, pi=pi)
"""
Description of the arguments
classifier (int): 0 if it is a classification problem else 1 if it is a regression problem
S (list[list]): list containing the index of variables S for each observation
pi (double): the level of the SDP, default value = 0.9
t (double): the radious of the ball of the SDP 

sdp_index[i, :size[i]] corresponds to the index of the variables in $S^\star$ of observation i  
sdp[i] corresponds to the SDP value of the $S^\star$ of observation i
"""

# Plotting of the Rule of observation 0 
rule = rules[0]
columns = [x_train.columns[i] for i in range(x_train.shape[1])]
rule_string = ['{} <= {} <= {}'.format(rule[i, 0] if rule[i, 0] > -1e+10 else -np.inf, columns[i],
                                       rule[i, 1] if rule[i, 1] < 1e+10 else +np.inf) for i in S]
rule_string = ' and '.join(rule_string)
```

Please find the experiments of the paper [HERE](https://github.com/aistats2022exp/ConsistentExplanations/tree/main/notebook)