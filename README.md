# Sign Language Detector using OpenCV and Python

This project utilizes OpenCV and Python to detect and classify hand gestures, enabling real-time interpretation of sign language. The program captures video input from the camera, detects hand gestures, and predicts the corresponding sign based on trained classification models.

## Installation and Setup

To run the Sign Language Detector, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/nrnavaneet/SignLanguageDetector.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained classification model (`keras_model.h5`) and label file (`labels.txt`) and place them in the `Model` directory.

## Usage

Run the `sign_language_detector.py` script to start the Sign Language Detector:

```bash
python sign_language_detector.py
```

The program will initialize the camera and display the live video feed. As you make hand gestures in front of the camera, the detector will recognize and classify them, displaying the corresponding sign on the screen.

To exit the program, press the 'q' key.

## Project Structure

The project structure is organized as follows:

- Model: Contains the pre-trained classification model and label file.
- sign_language_detector.py: Main Python script for detecting and classifying sign language gestures.
- cvzone: External module for hand tracking and classification (included in the repository).

## Contributions

Contributions to the project are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.

## Credits

This project utilizes the following libraries and frameworks:

- OpenCV: A computer vision library for image and video processing.
- cvzone: A Python library for hand tracking and classification.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README file with additional information about your project, installation instructions, usage guidelines, and more. Happy coding!
