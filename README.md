## Installation

Python version: 3.9

requirement.txt is provided in each folder, cd into the target folder and run the following command for installing the packages.

```bash
pip install -r requirements.txt
```

## Prompt 1
The code provides a raw solution for butterfly counting and doesn't achieve fly-in-fly-out detection. The solution uses frame differencing to identify the initial ROI, then uses the histogram of the ROI to conduct back projection to develop a new ROI for butterfly segmentation. 

Please run the following command in the prompt_1 folder to execute the code. A pop up window should display the video with the butterfly counts and the detected ROI.

```bash
python prompt_1.py
```

## Prompt 2

Only limited lines of code are implemented for sound wave exploration in a Jupyter Notebook.