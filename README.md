# **Keyword Spotter: Voice Activated Assistant**

## **Overview**

In this project, we aim to build a voice-activated assistant capable of recognizing and responding to specific keywords or commands using the Google Speech Commands dataset. Voice assistants rely on keyword spotting (KWS) to process users' spoken commands, typically activated by keywords such as "Alexa" or "Ok Google", which must be spotted to activate the voice assistant. This project explores the critical task of keyword spotting through speech recognition, using both machine learning and deep learning techniques. By converting raw audio waveforms into Mel Frequency Cepstral Coefficients (MFCCs), we create features that serve as input to our model, enabling efficient and accurate keyword detection.

### Built With

This project was built with

* python v3.10
* tensorflow v2.15
* The list of libraries used for developing this project is available at [requirements.txt](requirements.txt).

### Geting Started

To set up the environment and run the project, follow these steps:

#### Prerequisites

Ensure you have Python 3.10 installed. You can download it from [Python's official website](https://www.python.org/downloads/).

#### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Hamza-cpp/Keyword-Spotter-Voice-Activated-Assistant.git
    cd Keyword-Spotter-Voice-Activated-Assistant
    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

#### Running the Application

Run the following command to start the application:

```bash
python main.py
```

### Project Structure

* `main.py`: The main script to run the application.
* `requirements.txt`: List of required Python libraries.
* `saved_model.keras`: The saved model in TensorFlow's new format.

### Troubleshooting

#### Common Issues

1. TensorFlow Version Mismatch:

    Ensure you are using TensorFlow v2.15.0 for training for implementation. If you encounter any compatibility issues, consider aligning both environments to the same TensorFlow version.

    **To install TensorFlow v2.15.0:**

    ```bash
    pip install tensorflow==2.15.0
    ```

2. CUDA Drivers Not Found:

    If you see warnings related to CUDA drivers, it means your setup is missing GPU support. The application will run on CPU, which is slower. Ensure you have the correct CUDA and cuDNN versions installed if you need GPU support.
3. ALSA Library Warnings:

    If you see warnings related to ALSA library, it usually pertains to the audio backend and can often be ignored unless you encounter audio processing issues.
