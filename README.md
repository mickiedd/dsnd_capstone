[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/Screenshotfrom2019-04-2010-50-18.png "ScreenShot"

## File Overview

- dog_app.ipynb -> This the notebook that contain all the project codes.

- app/run.py    -> This is the Flash app for this project.

## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project in the AI Nanodegree! In this project, you will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, your algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output][image1]

Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app.  Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.  Your imperfect solution will nonetheless create a fun user experience!

## Project Instructions

### Instructions

- Theres no dog images data in this repo (cause it's to big to upload), so you need to copy it from Udacity repo.
 
- You must location youself in the root folder and type 'python app/run.py' to run the Flask app for this project.

### App Overview

![Sample Output][image4]

### Result

| Methods                     | Accuracy | Total Params |
|-----------------------------|----------|--------------|
| Simple CNN                  | 2.2727%  | 19,189       |
| VGG16 Transfer learning     | 38.6364% | 68,229       |
| Inception Transfer learning | 78.5885% | 272,517      |


- The simple 8 layers CNN architecture's Test accuracy: 2.2727%

- The VGG16 pre-trained bottleneck features Test accuracy: 38.6364%

- The InceptionV3 pre-trained bottleneck features Test accuracy: 78.5885%

### Conclusion

- The InceptionV3 pre-trained bottleneck features one that use transfer learning is much better than the other two, especially the first one, the simple 8 layers CNN architecture.

- It seems that, the more params it has, the more accuracy the model will get.
