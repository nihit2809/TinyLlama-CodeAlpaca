# Project README

This repository contains code for fine-tuning and deploying a conversational AI model based on the TinyLlama architecture using the CodeAlpaca dataset. The trained model is then deployed using FastAPI for real-time inference.

## Training

The training code is provided in the `train.py` file. Here's an overview of the steps involved:

1. **Setup**: Import necessary libraries and set up the environment, including CUDA memory management.
2. **Data Loading**: Load the CodeAlpaca dataset using the Hugging Face `datasets` library.
3. **Model Configuration**: Configure the TinyLlama model for fine-tuning. This includes defining quantization parameters, setting up the model architecture, and tokenizer.
4. **Training Arguments**: Specify the training arguments such as output directory, number of epochs, batch size, optimization algorithm, and learning rate scheduling.
5. **Training**: Initialize the SFTTrainer and start the training process. After training, save the fine-tuned model and tokenizer.

## Inference

The inference script is split into two parts: `infer.py` for the inference logic and `main.py` for setting up the FastAPI server. Here's how it works:

1. **Setup**: Import necessary libraries and set up the environment, including CUDA memory management.
2. **Model Loading**: Load the trained model and tokenizer for inference.
3. **FastAPI Setup**: Define the FastAPI application and configure the route for generating responses.
4. **Inference Logic**: Define a function to generate responses using the loaded model and tokenizer. The input question is processed, and the model generates a response based on the prompt.
5. **API Endpoint**: Create an endpoint to accept POST requests with input questions and return generated responses.

## Requirements

The `requirements.txt` file lists all the necessary Python libraries and their versions required to run the code.

## Usage

To train the model:

```
python train.py
```
To start the FastAPI server:
```
uvicorn main:app --reload
```


## Dependencies

Ensure that you have Python 3.x installed along with the required libraries listed in `requirements.txt`. Additionally, make sure to have CUDA enabled if using GPU for training and inference.

## Contributors

- [Nihit Agarwal] - [nihit6129@gmail.com]

## License

This project is licensed under the [License Name] License - see the LICENSE.md file for details.

Feel free to reach out with any questions or suggestions!

