# QAOA Angles

In the directory `QAOA_angles/`

There are 24 sets of angles in total. 

There are two text files which contain all of these angles. They can be parsed using python like this:

```
import ast
file = open("16.txt", "r")
angles = ast.literal_eval(file.read())
file.close()
```

Each text file contains one dictionary. This dictionary has 12 keys, named as 0-9 and then "pos." and "neg."

These keys describe the instances which resulted in these trained sets of angles. We want to keep track of which set of angles was used to generate which set of simulation data,
something to the effect of 16-0, 16-pos., 27-5, 27-neg., etc should work perfectly fine. 

The values in each dictionary are a list. This list is a contiguous set ("QAOA schedule") of angles that start at p=1 and go up to some p which changes depending on the instance. For all of the 27.txt instances, the angles only go to p=7. 

So, using python indexing, `angles[0]` is p=1, and so forth. 

The betas and gammas can be extracted from the lists like this in python:
```
beta = angles[:len(angles)//2]
gamma = angles[len(angles)//2:]
```


# Problem instance definitions

In the directory `Ising_models/`

There are exactly three problem instances. 
Each instance is given in two different formats, hopefully one or both will work for everyone to parse these in the various required formats. 
The first is a python dictionary written as a text file, which can be parsed using this code:

```
import ast
file = open("ibm_fez_0.txt", "r")
instance = ast.literal_eval(file.read())
file.close()
node_coeffs = instance[0]
edge_coeffs = instance[1]
triple_coeffs = instance[2]
```

The second has saved this same data structure as a matlab compatible file (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html)

- 127 var instance.
	- `ibm_kyiv_0.txt`
	- `Hamiltonian_ibm_kyiv_0.mat`
- 133 var instance
	- `ibm_torino_0.txt`
	- `Hamiltonian_ibm_torino_0.mat`
- 156 var instance
	- `ibm_fez_0.txt`
	- `Hamiltonian_ibm_fez_0.mat`
