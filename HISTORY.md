# History

### Framework Improvements
February 10th, 2025 | By Thomas Breimer | PR approved by Thomas Breimer 
- Can save simulations as videos
- Can decide to run in headless, video, window modes from terminal
- Can run individuals
- Can save individuals to videos
- Added some plots

Feb 11th | By Abhay Kashyap
- Added script to transform CMA-ES output to SNN weights and biases
- Script reshapes the flat array into an array of shape (NUM_SNN * PARAMS_PER_SNN) 
and then returns a list of dictionaries explicitly separating weights and biases of each SNN

### Introducing snn to evogym pipeline
Thurs. Feb 13th | Luodi Wang | SNN output to evogym through .json file
- Saving the output into a json file from a mock (random) input
- Deleted unused files: node.py, layers.py, net.py

### Command Line Args Improvements
February 13th, 2025 | By James Gaskell & Thomas Breimer
- Added proper named command line arguments for all scripts
- Improved implementation with argparse


### John's SNN Code
February 13th, 2025 | By Atharv Tekurkar
- Removed unecessay elements
- Fixed the functions that are giving errors
- Added unit tests for the functions

### Refactoring and Fixes
Feburary 15, 2025 | By Luodi Wang (partnered with Miguel Garduno on pylint)
- `get-output-state()` method in the snn class from a py file
- Within the SNN folder increased pylint score from 5.54/10 to 10/10 by 80.5%!


Feb 16th | Abhay Kashyap
- Made the `snn_parameters` dictionary elements compliant with the way SNN's `set_weights()` method works
- Completed testing unpacking tests with dummy cmaes output and setting them to SNNs

### Refactoring 
Feb 17th | Luodi Wang
- Successfully completed robot configuration with its locations of corners from bestbot.json
- Run inputs through the SNN
- and we've now finally outputted actuator control values


Feb 17th | Abhay Kashyap
- Merged all utility functions for SNN (eg. cmaes output => snn w&b; getting snn outputs, etc.) into one class
- Updated SNN code to fix some duty cycle and firelog errors (duty cycle was always 0 as firelog was not getting updated because of threshold)

Feb 23rd | Abhay Kashyap
- Removed files that were not being used and renames some files to more appropriate names
- Refactored `SNNRunner()` into `SNNController()` and the class simpler to use for simulation - removed redundant functions and converted some functions into private functions for abstraction.

Feb 24th | Thomas Breimer
- Completed integration

### SNN Firing Experiments
Feb 25th | By James Gaskell & Matthew Meek
- Added code to isolate a voxel and record the distances between robot corners
- Output corner values for each time step to csv for analysis
- Graphed expansion and contraction behaviors on a single voxel and large robot to determine SNN firing frequency

Feb 27th | Thomas Breimer
- cmaes bug fixes

Mar 4th | Thomas Breimer
- Added bestbot.json to snn-sim
- Chnaged run_cmaes.py to use bestbot.json

Mar 5th | Abhay Kashyap
- Created a log to track the frequency of target lengths obtained from SNN
- Added notebook showing a simple visualization of this frequency
- Added code to normalize inputs for better SNN computations