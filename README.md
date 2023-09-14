<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_torchvision_resnet/main/icons/pytorch-logo.png" alt="Algorithm icon">
  <h1 align="center">infer_torchvision_resnet</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_torchvision_resnet">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_torchvision_resnet">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_torchvision_resnet/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_torchvision_resnet.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

ResNet inference model for image classification. 

![Cat classification](https://raw.githubusercontent.com/Ikomia-hub/infer_torchvision_resnet/feat/new_readme/icons/output.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

- **model_name** (str) - default 'resnet18': Name of the pre-trained model. Additional ResNet size are available: 
    - resnet18
    - resnet34
    - resnet50
    - resnet101
    - resnet152

- **input_size** (int) - default '224': Size of the input image.
- **model_weight_file** (str, *optional*): Path to model weights file.
- **class_file** (str, , *optional*): Path to text file (.txt) containing class names. (If using a custom model)

**Parameters** should be in **strings format**  when added to the dictionary.

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_torchvision_resnet", auto_connect=True)

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(algo.get_image_with_graphics())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_torchvision_resnet", auto_connect=True)
algo.set_parameters({
    "input_size": "224",
})

# Run directly on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Inspect your result
display(algo.get_image_with_graphics())

```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_torchvision_resnet", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-dev/notebooks/main/examples/img/img_cat.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

