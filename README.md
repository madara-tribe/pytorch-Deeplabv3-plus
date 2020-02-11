# pytorch-Deeplabv3-plus's performance

<b>voc image1 (image/ overlay / segmentation)</b><hr>

![image1](https://user-images.githubusercontent.com/48679574/74228714-5df88f00-4d04-11ea-898d-7b0a0dd67375.png)

<b>voc image2 (image/ overlay / segmentation)</b><hr>

![image2](https://user-images.githubusercontent.com/48679574/74228918-bcbe0880-4d04-11ea-8c92-36eb8e41705c.png)


<b>voc image3 (image/ overlay / segmentation)</b><hr>

![image3](https://user-images.githubusercontent.com/48679574/74228936-c47dad00-4d04-11ea-82bd-7bccfe8b48a9.png)

<b>original image (image/ overlay / segmentation)</b><hr>

![original_image](https://user-images.githubusercontent.com/48679574/74244195-652f9500-4d24-11ea-95ad-74bd5fc6b6d9.png)


# How to train, mask annotation pixels, masking color type and accuracy

<b>How to train</b>

It is almost equal to fcn segmentation but deeplabv3plus network structure is almost different

・Activation => Relu

・Last layer shape => (None, H, W, 3)

・Jpg image is loaded by RGB

・Png mask image is loaded by grayscale

<b>mask annotation pixels</b>

Its is completely equal to fcn segmentation mask image

```If voc case, its num classes are 21 and its pixels range from 0 to 21```

<b>masking color type</b>

Mask color type is equal to its indexmap color.

<b>accuracy</b>

Its accuracy is more than fcn segmentation (training time for good accuracy is longer than fcn)
