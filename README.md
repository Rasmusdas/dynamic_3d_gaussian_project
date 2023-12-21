# Dynamic Pose Integration into Static 3D Gaussian Environments using Sparse Camera Setup
In this project we aim to recreate static scenes and add dynamic people into the scene using a single camera and recording the same action from multiple angles.

In the notebook you'll find out how to render the representations and how they are trained.

Here is a [link](https://drive.google.com/drive/folders/1jCHJ-p2XZ9G_YfjK8zg2acvkTjOCiYOD?usp=sharing) to the datasets if you want to train your own model. They also include a trained model.

Here are also some videos of both the static and dynamic scenes that were created.

[Double static composite](https://drive.google.com/file/d/18kkBAwH7lkDuBqbhszvDfKwbNQfevW1K/preview)

[Static with dynamic composite](https://drive.google.com/file/d/12Oh_4Zxk8LuQjZoeZhoGy7Y8x5aF4USe/preview)

[Dynamic without background](https://drive.google.com/file/d/1rJ09Zou75WvfTbCBbnYSOj-qAtoCydew/preview)

## Dependencies
Most of the dependencies can be installed with the included environment file, there are however 1 extra dependency that must be installed separately
```
git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
pip install .
```

## Acknowledgements
The code in this project is mostly a fork from [Dynamic3DGaussians](https://github.com/JonathonLuiten/Dynamic3DGaussians)
