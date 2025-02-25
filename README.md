
# Toolwear-Detection Project

## Project Introduction
**Toolwear-Detection** is a Python-based project designed to detect tool wear using machine learning techniques. The project categorizes tool wear into **fine, mild, and severe** classes.

## Image Classification
- **Data Transformation:** Preprocess images with resizing, normalization, and augmentation.
- **Model Configuration:** Modify a pre-trained **GoogLeNet**, **resnet** model to classify tool wear into three categories.
- **Classification:** Use the trained model to predict the class label of each image.

## Neural Networks
- **GoogLeNet:** Utilizes a pre-trained **GoogLeNet** model, fine-tuned for tool wear detection.
- **Custom Classifier:** Replaces the final fully connected layer to output three class probabilities.
- **Training and Evaluation:** Trained with **cross-entropy loss** and evaluated based on **accuracy, precision, recall, and F1 score**.

  ![Discriminator and Generator Structure](https://github.com/aryanpola/Toolwear-Detection/blob/main/nn_structure.jpg?raw=true)

### Generated Image
![Generated Toolwear_Image]([https://github.com/aryanpola/Toolwear-Detection/blob/main/nn_structure.jpg?raw=true](https://github.com/aryanpola/Toolwear-Detection/blob/main/generated_toolwear_image.png?raw=true))


## Usage
To use the tool wear detection algorithm, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/aryanpola/Toolwear-Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Toolwear-Detection
    ```
3. Install Dependencies:
   ```bash
    pip install -r requirements.txt
    ```
3. Run the main script:
    ```bash
    python main.py
    ```


## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/aryanpola/Toolwear-Detection/blob/main/LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue.

## Acknowledgments
Special thanks to all contributors and the open-source community for their invaluable support and contributions.

## Contact
For any inquiries or support, please contact me at aryanpola1603@gmaiil.com.

[GitHub Repository](https://github.com/aryanpola/Toolwear-Detection)
