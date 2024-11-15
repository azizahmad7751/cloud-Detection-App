# Cloud Detection App

A web application built with Flask that allows users to upload satellite images for cloud detection. Using image processing techniques and a pre-trained model, the app provides a prediction of cloud presence along with a confidence score.

## Features

- **Upload Satellite Images**: Upload images for cloud detection.
- **Cloud Prediction**: Get predictions on cloud presence along with a confidence percentage.
- **Contact and About Pages**: Learn more about the app and reach out to us.

## Demo

### Home Page
![Home Page](static/images/Home_Page.jpg)

### Upload Page
![Upload Page](static/images/Upload_Page.jpg)

### Results Page
![Results Page](static/images/result_Page.jpg)

## Technologies Used

- **Flask**: Backend web framework
- **OpenCV**: For image processing
- **NumPy**: For numerical operations
- **HTML/CSS**: For front-end design

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cloud-detection-app.git
   cd cloud-detection-app
2. **Install the required packages:**:
   ```bash
   pip install -r requirements.txt
3. **Run the app:**:
   ```bash
   python app.py
4. **Open the app**:
   -Visit in your web browser  
   ```bash
   http://127.0.0.1:5000

## Project Structure
- app.py: The main application file containing all Flask routes and logic for the cloud detection app.
- templates/: Directory for HTML templates, including index.html, upload.html, results.html, contact.html, and about.html.
- static/images/: Folder where uploaded images are stored for processing and display.
- static/css/: Stylesheets used for styling the web pages.
- model/modF6.h5: The pre-trained model file for cloud detection (handled via Git LFS or external hosting).
- requirements.txt: A list of required Python packages for the app.

## Usage
- Navigate to the Upload page to select and upload a satellite image.
- The app will preprocess the image and provide a prediction on whether clouds are present, along with a confidence score.
- View results and, if desired, upload additional images for further analysis.

## Customization
- Model: The predict_cloud function in app.py currently uses the modF6.h5 model. Replace it with another pre-trained model if necessary for improved accuracy.
- Styling: Modify HTML and CSS files located in the templates/ and static/css/ directories to personalize the app's look and feel.

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to report an issue, feel free to:
-Submit a pull request.
-Open an issue in the repository.

## Contact
-For questions, feedback, or support, please contact:
-Email: caps7751@gmail.com
