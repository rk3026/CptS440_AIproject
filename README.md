
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

Here's a video demo showing how to run the project: [https://youtu.be/S6UZpd5heYI](https://youtu.be/S6UZpd5heYI)

#### Required Files
- **Yelp Academic Dataset**: Download it [here](https://business.yelp.com/data/resources/open-dataset/) and place it in the `data` folder.
- **t5-emotions Model**:  
  To run the web app, you'll need the `t5-emotions` model. You can either:
  - Fine-tune it yourself by running `T5Finetune.py`, or
  - Download a pre-trained version [here](https://www.dropbox.com/scl/fi/1zavrwt2p2iks8uofz2jc/model.safetensors?rlkey=vtgs69mbrx571e9zn4haampiw&st=97aqoh3q&dl=0).

#### Web App
To launch the web application:
1. Navigate to the `web_app` folder.
2. Run `app.py`.
3. The app will start on a local server — open your browser and go to the address shown in the terminal.

#### Model Evaluation
To evaluate the AI models:
1. Go to the `backend` folder.
2. Run `model_evaluation.py`.

This will generate several PDFs and images inside the `evaluation` folder.

#### Other
Additional Jupyter notebooks for testing are included in the repository. You can run them after setting up the environment.

