# Handwritten-Character-Recognition

Steps of processing the image and identifying the digit: 

1. Deskewing (Pre-processing)
Aligning digits before building a classifier produces superior results. In the case of faces, alignment is rather obvious — you can apply a similarity transformation to an image of a face to align the two corners of the eyes to the two corners of a reference face.

![image](https://user-images.githubusercontent.com/48985829/139740570-9d06691a-a099-4a92-863b-14816968a345.png)

2. Calculate the Histogram of Oriented Gradients (HOG) descriptor:

=> Preparing our Dataset
    • How we prepare the data?
        500 samples of each digit with 5 rows of 100 samples
    • Each character is a grayscale 20 x 20 pixels
    • We use numpy to arrange the data in this format:
        50 x 100 x 20 x 20
    • We then split the training dataset into 2 segments and flatten our 20x20 array.
    • Training Set - 70% of the data.
    • Test Set - 30% of the data - we use a test set to evaluate our model.
    • Each dataset is then flattened, meaning we turn the 20 x 20 pixel array into a flat 1x400. Each row of 20 pixels is simply appended into one long column. 
    • We then assign labels to both training & test datasets (i.e. 0,1,2,3,4,5,6,7,9).

3. About the Digits Data Set

=> The digits.png image contains 500 samples of each numeral (0-9).
=> Total of 5000 samples of data.
=> Each individual character has dimensions: 20 x 20 pixels.

![image](https://user-images.githubusercontent.com/48985829/139740753-eba6e18e-e797-4ac4-8bda-fbcb16985836.png)

4. Pocessing steps: 

![image](https://user-images.githubusercontent.com/48985829/139740844-cf80a995-74ed-413a-8800-662d30b04f43.png)

5. END-RESULT:

![image](https://user-images.githubusercontent.com/48985829/139740930-257a9b6e-17da-4e4d-8bc3-a45936d1d0cc.png)


