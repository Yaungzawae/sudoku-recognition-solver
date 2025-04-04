# Sudoku Recognition Solver [from Scratch (kinda)]
## by YaungZawAe And XavyWavy

This project consists of three main components:

- A neural network for handwriting recognition
- Image preprocessing
- An algorithm to solve the Sudoku puzzle

### Neural network
inspired by [Samson Zhang](https://youtu.be/w8yWXqWQYmU?si=O_Xk4FOacG5kkrck)

For this neural network, we use a 2-layer MLP (Multi-Layer Perceptron). 
The structure goes as follow: \
    Input → First Dense Layer(784 → 9, Linear Transform) → ReLU Activation ->
    Second Dense Layer (9 → 9, Linear Transform) → Softmax Activation → Loss Calculation
    -> Backpropagation & Gradient Descent Optimization

Input Layer is just a list of length 784, each representing each grayscale pixel of the image from 0.0 (white) to 1.0 (black). Then it is passed into the dense layer.

The Dense Layer receives the 784 inputs and transforms them into a list of length 9, representing raw scores for each class. But we do not stop here—we then pass these scores into the ReLU function 𝑓(𝑥) = max(0,𝑥), which removes negative values and introduces non-linearity for solving complex problems. After another dense layer, we apply the Softmax function, which converts the final outputs into probabilities.






### Image Preprocessing
In week 10, we explored Hough lines, a technique for detecting straight lines in noisy data. This method is particularly useful for our task of extracting handwritten digits from a Sudoku grid.

To recognize each digit, we need to isolate it in a 28×28 pixel square, free from noise. To achieve this, we draw 10×10 Hough lines over the image, effectively segmenting it into 9×9 individual squares, each containing a single digit.

However, drawing Hough lines presents challenges due to the high resolution of camera images. Since a single black line may be detected multiple times, the process becomes computationally expensive. To mitigate this, we first reduce the image resolution to 600×600 pixels before applying the Hough transform.

Next, we apply a Canny edge filter (not yet implemented) to detect edges, which converts each black line into two separate edges. Hough lines are then drawn based on these edges, resulting in two detected lines for each actual black line.

To refine the results, we filter out unnecessary and incorrect lines based on distance, angle, and minimum spacing criteria. The final lines should be strictly horizontal and vertical, ensuring they do not intersect improperly.

## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

