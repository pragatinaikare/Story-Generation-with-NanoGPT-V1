## Story Generation with NanoGPT


This project involves developing a nanoGPT language model from scratch to generate stories using the Tiny Stories dataset from Hugging Face. The nanoGPT architecture, a lightweight version inspired by the main GPT model, was used in this project and comprises 28 million parameters.

## Dataset

- **Dataset**: [Tiny Stories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Number of Stories**: 2.12M
- **Tokenizer Used**: GPT2 Tokenizer
- **Generating a Batch**: Selected 100 stories randomly. Added start and end token to distinguish stories and selected a window context length size tokens randomly.

## Training 
Replicated the pretraining phase of LLM to generate a story. ​


Trained a small language model of 28Million parameters ​


Trained on predicting next token in the vocabulary of size 50k​

| **Size** | **Layers** | **Embedding Dimension** | **Attention Heads** | **Tokens Trained** | **Context Length** |
|----------|------------|-------------------------|---------------------|--------------------|---------------------|
| 28M      | 4          | 256                     | 8                   | 1.2B               | 512                 |

​
​
## Evaluation
Model: nanoGPT (28.02 Million parameters)

Training Loss: 2.0

Validation Loss: 1.88

![Alt text](Images/lossSnap.png)
## Output Snapshots 

![Alt text](Images/OutputSnap.png)


## Potential Improvements 
If you can see in the output, stories are half meaningful. Due to resource constraints, I couldn't run the big model, but with proper resources, we can try running a bigger model to get more meaningful stories.

