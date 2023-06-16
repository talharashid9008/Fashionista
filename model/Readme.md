# FYP 1 Code

image segmentation using U-net

## Installation
* Open Anaconda Terminal 
```bash
conda create -n dl python=3.8 anaconda
```
* Above command will create the environment
* Now install Pytorch:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install OpenCV.

```bash
pip install opencv-python
```

## Usage
* For each Image in input_images directory transform and normalize, then pass it from simple U-NET model.
* By applying color palette we are giving different color to different segments.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)