# Handwritten-Character-Recognition

Steps of processing the image and identifying the digit: 

1. Deskewing (Pre-processing)
Aligning digits before building a classifier produces superior results. In the case of faces, alignment is rather obvious — you can apply a similarity transformation to an image of a face to align the two corners of the eyes to the two corners of a reference face.

img-1

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

img-2

4. Pocessing steps: 

img-3

5. END-RESULT:

img-4


