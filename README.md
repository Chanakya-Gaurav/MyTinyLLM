# MyLLM
This is a tiny LLM for Education Purpose


## Execution

### BiGram Code Walkthorugh
- Demonstrate the learning process using demo_tinyLLM.ipynb Notebook

### BiGram Execution
- Demonstrate the execution time difference and the accurayc using the following commands

- Training speed with GPU and CPU
- Try prompts like 't' or 'h'
- Notice the wrong next character for lesser training iterations
- Notice the correct next character prediction for training iterations - but still gibbrish following it 

1. Using CPU - lesser training iterations
- python .\tinyLLM.py --load_train 0 --iters 100 --use_gpu 0

2. Using CPU - higher training iterations
- python .\tinyLLM.py --load_train 0 --iters 3000 --use_gpu 0

3. Using GPU - same higher iterations - training time dropped significantly
- python .\tinyLLM.py --load_train 0 --iters 3000 --use_gpu 1

4. Load pre-trained model - 30K iterations - better prediction
- python .\tinyLLM.py --load_train 1

5. Load the pre-trained GPT model
- python .\interact.py


### Demonstrate the tinyGPT Execution
- Do a brief code wak through of tinyGPT.py
- First show the TransformerBlock
- Show how the TransformerBlock calls the MultiHead (which calls the Head) followed by a FeedForward
- Draw attention to Q, K, V defined in the Head block
- Show how to Compute attention scores
- Convert scores to probabilities with softmax
- Apply attention weights to values

- Show the expansion, activaition and contraction layers in the feed forward layer 
1. python .\interact.py
- Try prompts like 'hope t' or 'hope h'
- Try prompts like 'hello h' and 'hello w'