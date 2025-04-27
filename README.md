
# **CptS440_AIProject**

## Overview
This project focuses on **Sentiment Analysis** of text data.

It contains a web application where users can provide text/data to run sentiment analysis on. The web application will process the text data provided and then output a summary of the overall sentiment of that data through various graphical elements.
Behind the web app, we utilize various pretrained/fine-tuned transformer-based AI models to process text data.

## Setup Instructions

### 1. Clone the Repository
If you haven’t already, clone this repository to your local machine using Git:

```bash
git clone https://github.com/your-username/CptS440_AIproject.git
```

Then navigate to the folder:
```bash
cd CptS440_AIproject
```

### 2. Create Virtual Environment
Run the appropriate setup script from the `/scripts` folder based on your operating system:

- **For Windows:**
  ```bash
  setup.bat
  ```

- **For macOS/Linux:**
  ```bash
  setup.sh
  ```

This will create a **virtual environment** with all the necessary dependencies listed in `requirements.txt`.

### 3. Running the Project
Here is a video demo of running the project: https://youtu.be/S6UZpd5heYI

Required Files:
Download the Yelp Academic Dataset and place in the data folder: https://business.yelp.com/data/resources/open-dataset/

We also need to have the t5-emotions model for the web app to run. To get this model, run the T5Finetune.py file, or download a version here: https://www.dropbox.com/scl/fi/1zavrwt2p2iks8uofz2jc/model.safetensors?rlkey=vtgs69mbrx571e9zn4haampiw&st=97aqoh3q&dl=0

#### Web App
To launch the web application, navigate to the `web_app` folder and run `app.py`. The app will start on a local port — open your browser and go to the displayed address to access it.

#### Model Evaluation
To evaluate the AI models, go to the `backend` folder and run `model_evaluation.py`. This will generate multiple PDFs and images inside the `evaluation` folder.

#### Other
Additional Jupyter notebooks used for testing are available in the repository. You can run them after completing the environment setup.

