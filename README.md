# Sudoku Recognition Solver
## by Ye Zaw Aung and Min Tuta Naing

This project consists of three main components:

- A neural network for handwriting recognition
- Image preprocessing
- algorithm to solve the Sudoku puzzle

### Neural network

Inspired by [Samson Zhang](https://youtu.be/w8yWXqWQYmU?si=O_Xk4FOacG5kkrck), I designed a simple two-layer MLP (Multi-Layer Perceptron) for recognizing digits from Sudoku images.

Model architecture:

Input (784) → Dense (784 → 9) → ReLU → Dense (9 → 9) → Softmax → Cross-Entropy Loss

The input is a 28×28 grayscale image flattened into a 784-length list.

The first dense layer outputs raw scores for each digit class (1 to 9).

A ReLU activation introduces non-linearity.

The second dense layer is followed by Softmax, converting scores to probabilities.

The predicted class is the one with the highest probability.

I train the model using cross-entropy loss and optimize it with backpropagation and gradient descent.

Note that, I excluded the zeros in the training data because sudoku does not include 0.

Performance:

~97% accuracy on training data

~94% on MNIST test set

~80–85% on real handwriting, likely due to differences in stroke thickness, lighting, and handwriting style


### Image Preprocessing
In week 10, I explored Hough lines, a technique for detecting straight lines in noisy data. This method is particularly useful for our task of extracting handwritten digits from a Sudoku grid.

To recognize each digit, I need to isolate it in a 28×28 pixel square, free from noise. To achieve this, drawing 10×10 Hough lines over the image, effectively segmenting it into 9×9 individual squares, each containing a single digit.

However, drawing Hough lines presents challenges due to the high resolution of camera images. Since a single black line may be detected multiple times, the process becomes computationally expensive. To mitigate this, we first reduce the image resolution to 600×600 pixels before applying the Hough transform.

Next, apply Canny edge filter to detect edges, which converts each black line into two separate edges. Hough lines are then drawn based on these edges, resulting in two detected lines for each actual black line.

To refine the results, filter out unnecessary and incorrect lines based on distance, angle, and minimum spacing criteria. The final lines should be strictly horizontal and vertical, ensuring they do not intersect improperly. I can calculate the coordinates of the box by finding the points horizontal and vertical lines meet each others.

Then I get squares (not 28 * 28 yet) of pixels which has either black space or handwritten digits. We can figure out which cell is empty by finding the sum of the pixel values and then thresholding. For recognizing the numbers, I used our neural network. For that I resize the squares into 28*28 to pass into our model.

At first, I get around 40% successful prediction from just resizing the image alone. Then, I learnt that MINST dataset is centered and has 4px padding, so I try to format our boxes. I zoom in the handwriting(white pixels) and fit it into 20*20 (4 padding each side excluded) and then I paste into the black 28 * 28 board. When I apply it into the our handwritten data, the accuracy goes up to 55-60%. When I tweak our code by applying the same formatting in the training process, it goes up to 80-85% which I ended up using in our final version.

### Solving the Sudoku puzzle
This is done by [Xavier-Naing](https://github.com/Xavier-Naing) which uses backtracking to solves the puzzle.

### Display interface
At first, we tried with gradio to make interactive interface but we couldn't make it interactive. When changing the cells manually of gradion data frame, it does not change the dataframe and only return the original one. Therefore, we switched to pygame.


###  🛠️ How to Run
To run the full Sudoku recognition and solving pipeline, use the following command:


```python main.py path_to_your_image```

Replace path_to_your_image with the actual path to your Sudoku image file. For example:

```python main.py images/sudoku1.png```

Load and preprocess the image

Use the trained neural network to recognize digits

Solve the puzzle using backtracking

Display the solved Sudoku using pygame

Make sure your image is a clear topdown photo of a handwritten Sudoku grid with visible lines and digits.
