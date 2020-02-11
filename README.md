# pytorch-Deeplabv3-plus

<b>voc image1 (image/ overlay / segmentation)</b><hr>

![image1](https://user-images.githubusercontent.com/48679574/74228714-5df88f00-4d04-11ea-898d-7b0a0dd67375.png)

<b>voc image2 (image/ overlay / segmentation)</b><hr>

![image2](https://user-images.githubusercontent.com/48679574/74228918-bcbe0880-4d04-11ea-8c92-36eb8e41705c.png)


<b>voc image3 (image/ overlay / segmentation)</b><hr>

![image3](https://user-images.githubusercontent.com/48679574/74228936-c47dad00-4d04-11ea-82bd-7bccfe8b48a9.png)

<b>original image (image/ overlay / segmentation)</b>

# How to train, mask annotation pixels and accuracy

<b>How to train</b>
It is almost equal to fcn segmentation but NN structure is almost different
・activation => relu
・last layer shape => (None, H, W, 3)
・jpg image is loaded by RGB
・png mask image is loaded by grayscale

<b>mask annotation pixels</b>
Its is completely equal to fcn segmentation mask image
```If voc case, its num classes are 21 and its pixels range from 0 to 21```


<b>accuracy</b>
Its speed is less than fcn segmentation, but accuracy is more than fcn segmentation
