# ArtLine

This repo has been forked from Vijish Madhavan (vijishmadhavan) and altered by Michael Sexton (MichaelSexton21) to be compatable with Code Ocean's App Panel Feature (https://codeocean.com/).

This project uses deep learning to create line art pictures. The original readme has been renamed to README_old.md and contains all of the deserved credit and acknowledgments.

## Code Ocean Setup
* Create a new capsule from this repository
* Select the environment:
  * Saas: "Python (3.8.1, miniconda 4.8.2)"
  * VPC: "Python (3.8.1, miniconda 4.8.2, jupyterlab 2.1.1)"
* Move all the files except this file (README.md) into the code folder
* Use the environment_requirements.txt file to build the environment
* Set Artline.py to run
* Use the two links below to download the models for 650 and 920 models to the code folder
  * I recommend launching a CW station (Terminal or JupyterLab) and using the code/download_models.sh script I wrote to help
  * You need to change the permission of the file to be able to run it using the commands:
     * chmod +755 download_models.sh
     * ./download_models.sh
* Open up the App Panel tab, you will need to add two parameters:
  * File parameter: Image File Path (default: lion.png)
  * List parameter: Output Image Size (920 or 650 )
* In the data/image/ folder, place what ever image you like to alter here
* Run the project by specifiy your desired settings in the App Panel and press "Run with parameters" to begin
* **[Warning] When you run the capsule you will recieve the following error. Please ignore it as the results do not change if you have a gpu or not.**
  *   "UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.)"    
* If you wanted to run it using the command line/run file:
  * First specify the relative path to the image from the code folder
   * Next specify the size of the output you want (650 or 920)
   * Example: python -u Artline.py "../data/images/lion.jpg" 650

You need to add a new file later when running the App Panel with your own input image.
The output image should be in the results folder as a file prefixed with "line..."

## Model Download Links
Because the origional repository was not initalized with Github's Large File Storage I cannot add that funcitonality after I forked the repository so I have linked the two pretrained models here:

**650 Pixel Output Model:** https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1

**920 Pixel Output Model:** https://www.dropbox.com/s/04suaimdpru76h3/ArtLine_920.pkl?dl=1

## Example Images

bohemian rhapsody movie , Rami Malek American actor

![bohemian](https://i.imgur.com/od6IA08.jpg)



Photo by Maxim from Pexels

![Imgur](https://i.imgur.com/yksAvUq.jpg)



Friends, TV show.

![Friends](https://i.imgur.com/x3vbPys.jpg)


Keanu Reeves, Canadian actor.

![Keanu](https://i.imgur.com/labkc8V.jpg)



Hrithik Roshan

![Hrithik](https://i.imgur.com/U1sktwM.jpg)

Alita: Battle Angel

![Alita](https://i.imgur.com/3gcBKq2.jpg)

Virat Kohli, Indian cricketer

![Virat](https://i.imgur.com/jg76waU.jpg)

Photo by Anastasiya Gepp from Pexels

![Imgur](https://i.imgur.com/xWEUK7W.jpg)

Interstellar

![Interstellar](https://i.imgur.com/xiuwDGd.jpg)

Pexels Portrait, Model

![Imgur](https://i.imgur.com/NMaPOiE.jpg)

Beyoncé, American singer

![Beyoncé](https://i.imgur.com/QalvHKS.jpg)

## Cartoonize

**Lets cartoonize the lineart portraits, its still in the making but have a look at some pretty pictures.**

Skrillex , American DJ

![Imgur](https://i.imgur.com/BJW8beC.jpg)

Tom Hanks, Actor

![Imgur](https://i.imgur.com/hvkDTZR.jpg)


## Line Art

The amazing results that the model has produced has a secret sauce to it. The initial model couldn't create the sort of output I was expecting, it mostly struggled with recognizing facial features. Even though (https://github.com/yiranran/APDrawingGAN) produced great results it had limitations like (frontal face photo similar to ID photo, preferably with clear face features, no glasses and no long fringe.) I wanted to break-in and produce results that could recognize any pose. Achieving proper lines around the face, eyes, lips and nose depends on the data you give the model. APDrawing dataset alone was not enough so I had to combine selected photos from Anime sketch colorization pair dataset. The combined dataset helped the model to learn the lines better.

## Movie Poster created using ArtLine.

The movie poster was created using ArtLine in no time , it's not as good as it should be but I'm not an artist.

![Poster](https://i.imgur.com/QuRnKjB.jpg)

![Poster](https://i.imgur.com/RvTTxdI.jpg)


## Technical Details

* **Self-Attention** (https://arxiv.org/abs/1805.08318). Generator is pretrained UNET with spectral normalization and self-attention. Something that I got from Jason Antic's DeOldify(https://github.com/jantic/DeOldify), this made a huge difference, all of a sudden I started getting proper details around the facial features.

* **Progressive Resizing** (https://arxiv.org/abs/1710.10196),(https://arxiv.org/pdf/1707.02921.pdf). Progressive resizing takes this idea of gradually increasing the image size, In this project the image size were gradually increased and learning rates were adjusted. Thanks to fast.ai for intrdoucing me to Progressive resizing, this helps the model to generalise better as it sees many more different images.

* **Generator Loss** :  Perceptual Loss/Feature Loss based on VGG16. (https://arxiv.org/pdf/1603.08155.pdf).

**Surprise!! No critic,No GAN. GAN did not make much of a difference so I was happy with No GAN.**

The mission was to create something that converts any personal photo into a line art. The initial efforts have helped to recognize lines, but still the model has to improve a lot with shadows and clothes. All my efforts are to improve the model and make line art a click away.

![Imgur](https://i.imgur.com/fhUi3uv.jpg)

## Dataset

[APDrawing dataset](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/APDrawingDB.zip) 

Anime sketch colorization pair dataset

APDrawing data set consits of mostly close-up portraits so the model would struggle to recogonize cloths,hands etc. For this purpose selected images from Anime sketch colorization pair were used.


## Going Forward

I hope I was clear, going forward would like to improve the model further as it still struggles with random backgrounds(I'm creating a custom dataset to address this issue). Cartoonizing the image was never part of the project, but somehow it came up and it did okay!! Still lots to improve. Ill release the cartoonize model when it looks impressive enough to show off.

*I will be constantly upgrading the project for the foreseeable future.*

## Getting Started Yourself

The easiest way to get started is to simply try out on Colab: https://colab.research.google.com/github/vijishmadhavan/Light-Up/blob/master/ArtLine(Try_it_on_Colab).ipynb

### Installation Details

This project is built around the wonderful Fast.AI library.

- **fastai==1.0.61** (and its dependencies).  Please dont install the higher versions
- **PyTorch 1.6.0** Please don't install the higher versions

### Limitations

- Getting great output depends on Lighting, Backgrounds,Shadows and the quality of photos. You'll mostly get good results in the first go but there are chances for issues as     well. The model is not there yet, it still needs to be tweaked to reach out to all the consumers. It might be useful for "AI Artisits/ Artists who can bring changes to the final output.

- The model confuses shadows with hair, something that I'm trying to solve.

- It does bad with low quality images(below 500px).

- I'm not a coder, bear with me for the bad code and documentation. Will make sure that I improve with upcoming updates.

### Updates

[Get more updates on Twitter](https://twitter.com/Vijish68859437)

Mail me @ vijishmadhavan@gmail.com

### Acknowledgments

- The code is inspired from Fast.AI's Lesson 7 and DeOldify (https://github.com/jantic/DeOldify), Please have look at the Lesson notebook (https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson7-superres-gan.ipynb)

- Thanks to (https://github.com/yiranran/APDrawingGAN) for the amazing dataset.

## License

All code in this repository is under the MIT license as specified by the LICENSE file.





