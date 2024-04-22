This repository shares two interactive Jupyter Notebooks to learn about the calibration of a lumped rainfall-runoff model (the GR6J model)
* **GR6J_AutCal_Notebook.ipynb** uses one year of data (precipitation, potential evapotranspiration and flow) from an example catchment, to begin to learn about the role of the different parameters, and how varying them changes the shape of the simulated hydrograph and a set of performance metrics
* **GR6J_OATAnalysis_Notebook.ipynb** applies model calibration to the same example catchment using a simple Monte Carlo simulation approach, whereby a prescribed number of possible parameter combinations are randomly generated and run through the model, and the one that maximise the chosen performance metric is selected.

The repository also includes the python code to simulate the GR6J model and to calculate performance metrics. More explanation and scientific references are embedded in the Notebooks and code.

<!-- How to launch Jupyter Notebooks from Anaconda Navigator:
 
Open Anaconda Navigator, click on "Jupyter Notebooks", this will automatically open a new tab in your web browser, in this tab you will see your folders and file, go to the folder where the Notebook is saved, click on it, and the Notebook will open in your browser.
![Alt Text](https://github.com/FrancescaPianosi/RRmodelling/blob/main/HowToLaunchJN.gif)

You can now run and edit it directly from your browser.
-->
