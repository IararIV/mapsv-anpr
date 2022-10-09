Data governance - how images are stored in our application so the next stage know where to look at.

images/\
|_ raw_images/\
    |_ (plate).jpg\
|_ localization/\
    |_ (plate)\_localization.jpg\
|_ segmentation/\
    |_ (plate)\_segmentation.jpg\
|_ ocr/\
    |_ (plate)\
        |_ chars/\
            |_ (index).jpg\
        |_ digits/\
            |_ (index).jpg\
|_ predictions\
    |_ (plate)\_prediction\_(predicted plate).jpg

We also have a folder 'cluster' with the images generated to train the model.
