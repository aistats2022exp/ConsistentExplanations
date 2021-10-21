# Consistent Sufficient Explantions and Minimal Sufficient Rules 
 
Active Coalition of Variables ACV is a Package that implemented the SDP Approaches of the Paper: [ref].
 
## Requirements
Python 3.6+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

## Installation

Install the acv package:
```
$ pip install acv-exp
```

## How does ACV work?
To compute the different explanations, we only need a **trained Random Forest** and 
- Data **(X, Y)** if we want to explain directly the data
- Or **(X, f(X))** if we want to explain the model f

### Examples:

```python
from acv_explainers import ACVTree

forest = RandomForestClassifier() # or  Random Forest Regressor models
#...trained the model

# Initialize the explainer
acvtree = ACVTree(forest, data) # data should be np.ndarray with dtype=double
```

Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we do not observe the variables <img src="https://latex.codecogs.com/gif.latex?x_{\bar{S}}" />.
* **How to compute  the Same Decision Probability of a subset S <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
sdp = acvtree.compute_sdp_rf(X, S, data)

"""
Description of the arguments    
   
X (np.ndarray[2]): observations        
S (np.ndarray[1]): index of variables on which we want to compute the SDP
data (np.ndarray[2]): data used to compute the SDP
"""
```
* **How to compute All the Sufficient Explanations <img src="https://latex.codecogs.com/gif.latex?S^\star" />** ?
```python 
sufficient_coal, sdp_coal, sdp_global = acvtree.sufficient_coal_rf(x_test, y_test, x_train, y_train, stop=False, global_proba=global_proba,
                                                                    classifier=0, t=t)

"""
Description of the arguments

classifier (int): 0 if it is a classification problem, and 1 if it is a regression problem
t (np.double): Only necessary for regression problem, it corresponds to the radius of the ball of the SDP 
"""
```

*  **How to compute the Minimal Sufficient Explanation ?**
```python
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C=[[]], global_proba=0.9)

"""
Description of the arguments

X (np.ndarray[2]): observations
data (np.ndarray[2]): data used for the estimation
C (list[list]): list of the index of variables group together
global_proba (double): the level of the SDP, default value = 0.9

sdp_index[i, :size[i]] corresponds to the index of the variables in $S^\star$ of observation i  
sdp[i] corresponds to the SDP value of the $S^\star$ of observation i
"""
```

* **How to compute the Minimal Local Rule based on a Sufficient Explanation S**

```python
sdp, rules, sdp_all, rules_data = acvtree.compute_sdp_maxrules(x_test, y_test, x_train, y_train,
                                                    S=[S], classifier=0, t=t, pi=pi)

# Rule of observation 0 
rule = rules[0]
columns = [x_train.columns[i] for i in range(x_train.shape[1])]
rule_string = ['{} <= {} <= {}'.format(rule[i, 0] if rule[i, 0] > -1e+10 else -np.inf, columns[i],
                                       rule[i, 1] if rule[i, 1] < 1e+10 else +np.inf) for i in S]
rule_string = ' and '.join(rule_string)
```
