# LLM

* 全面、通俗的入门教程 [Large Language Model Tutorial Series: 30 Step-by-Step Lessons [FREE\][2024] | by Ayşe Kübra Kuyucu | Tech Talk with ChatGPT | Medium](https://medium.com/tech-talk-with-chatgpt/large-language-model-tutorial-series-30-step-by-step-lessons-free-c8e6114b0c74)

## LLM 概念

* A **large language model (LLM)** is a computational model notable for its ability to **achieve general-purpose language generation and other natural language processing tasks** such as classification. Based on language models, LLMs acquire these abilities by **learning statistical relationships from vast amounts of text** during a computationally intensive self-supervised and semi-supervised training process.

  > from [Large language model - Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
  >
  > 在笔者看来，有两点需要明确。
  >
  > 1. LLM 从大量文本数据学习统计关系（因此，所有 LLM 表现出的智能如逻辑推理等都是基于大量文本得到的统计关系）
  > 2. LLM 的主要能力是通用语言生成和其他自然语言处理任务（对于 LLM 在其它任务上的表现分析，都应该联系到 LLM 在通用语言生成、自然语言处理的本职任务）

* LLM 推理的本质是以一种自然的方式生成延续初始提示的新文本。LLM 模型的本质是一个参数众多的神经网络。LLM 学习的本质是习得了自然语言的概率分布。LLM 模型文件的本质是是一个概率数据库，它能够为任何特定的字符以及其上下文相关的字符赋予一定的概率分布。

* LLM 的构建：

  * 预训练：LLM 训练背后的创新在于 Transformer 架构的引入，该架构使模型能够从大量数据中学习，同时保留输入不同部分之间的关键上下文关系。经过大量数据的预训练后，LLM 可以视作一个文档补全器，此时与 ChatGPT 这样的助理人工智能还存在一些区别。

    > 预训练的数据是海量的、高质量的，并且无需标注

  * 微调：基于有标注的指令，对模型进一步训练。这一步后，LLM 补全提示词的方式将会趋近训练数据的风格。

    > 微调的数据通常没有预训练的规模大，并且含有标注。但它们同样需要高质量。

  * RLHF：根据人类反馈进行强化学习。

  * 提示工程：可以仔细设计提示以从模型中获得所需的响应（有时甚至无需微调）。

    > LLM 根据初始提示，选择概率最高后续文本。而提示工程的本质就是通过优化初始提示，让那些期望回答的字符的概率更高，进而引导LLM行为朝着特定结果。

## 语言模型

> 语言模型是 LLM 发展的基础，很多 LLM 的概念都是在语言模型中已有的应用。

* 语言模型是许多自然语言处理（NLP）应用的核心，包括机器翻译、语音识别、文本摘要、问答系统和文本生成等。它们通过预测给定一系列文本中下一个词或词元，从而学习语言的统计规律，例如语法、语义和语用学。
* 语言模型是一种数学模型，用于为自然语言中的词或词元序列分配概率。这些序列可以是句子、段落或任何文本片段。序列的概率反映了它在自然语言中出现的可能性。
  * 目标是从大量文本数据中学习语言的统计规律。理解词和词元是如何关联的，如何构成有意义的句子和段落，以及如何传达信息和知识。
* 语言模型主要分为以下几种：
  * **N-gram模型**：通过考察序列中前 n-1 个词或词元来预测下一个词或词元的概率。
    * N-gram模型实现简单，训练迅速，但它们容易受到数据稀疏性和泛化能力不足的影响。数据稀疏性是指训练集中未观察到的词或词元序列，导致这些序列被赋予零概率。泛化能力不足是指 N-gram模型 无法捕捉距离较远的词和词元之间的长期依赖和语义关系。
  * **循环神经网络（RNN）模型**：RNN通过循环连接一次处理一个词或词元序列，并更新代表模型记忆的隐藏状态。隐藏状态随后用于预测序列中的下一个词或词元。
    * RNN模型克服了 N-gram模型 的数据稀疏性和泛化能力不足问题，因为它们可以学习任意长度和频率的词或词元序列。它们还可以捕捉长期依赖和词与词元之间的语义关系，因为隐藏状态可以存储整个序列的信息。然而，RNN模型也存在一些缺点，如训练长序列时的梯度消失问题，以及训练和推理的速度较慢。
  * **卷积神经网络（CNN）模型**：CNN 通过卷积操作对词或词元序列应用滤波器，提取表示语言模式和规律的局部特征。滤波器的大小和形状各异，能够捕捉不同层次和粒度的特征。
    * CNN模型 同样克服了 N-gram模型 的数据稀疏性和泛化能力不足问题，因为它们可以学习任意长度和频率的词或词元序列。它们还可以捕捉长期依赖和词与词元之间的语义关系，因为滤波器能够覆盖序列的大部分区域。此外，CNN模型 在训练和推理速度上相对于RNN模型具有优势，也更稳定和可靠。
  * **Transformer 模型**：是最新和最先进的用于语言建模的神经网络。变换器模型使用自注意力机制对词或词元序列进行编码和解码，并应用注意力机制来关注序列中最相关的部分。

## Transformer

首先瞻仰下开山之作、LLM万物起源**《Attention Is All You Need》**[^2]。尤其注意Transformer模型架构示意图：

![image-20240802170045154](./img/image-20240802170045154.png)

在深入了解LLM前，先通过一个简单示例[^1]形象理解Transformer。

* Before encoder: Calculating Word embedding and positional embedding:

  * The vocabulary size determines the total number of **unique words** in our dataset.

  * Encoding: assign a unique number to each unique word

  * Word embedding: uses a n-dimensional embedding vector for each input word

  * positional embeddings: There are two formulas for positional embedding depending on the position of the ith value of that embedding vector for each word.

    > input 中 一个 word 的 positional embedding 是由它的 embedding、它在 input 中的序列决定

  * resultant matrix: This resultant matrix from combining both matrices (**Word embedding matrix** and **positional embedding matrix**) will be considered as an input to the encoder part.

* Calculating Multi Head Attention: A multi-head attention is comprised of many single-head attentions. It is up to us how many single heads we need to combine.

  * a single-head attention: There are three inputs: **query**, **key**, and **value**. Each of these matrices is obtained by multiplying a different set of weights matrix from the **Transpose** of same matrix that we computed earlier by adding the word embedding and positional embedding matrix.

    > query/key/value matrix = weights matrix(query/key/value) * trans(Word embedding matrix + positional embedding matrix)

    1. MatMul: mat1 = query * key
    2. Scale: mat2 = mat1 / sqrt(the dimension of our embedding vector)
    3. Mask(opt.): It helps the model understand things in a step-by-step manner, without cheating by looking ahead.
    4. SoftMax: mat3 = softmax(each cell in mat2)
    5. the resultant matrix from multi-head attention = mat3 * value = $softmax(\frac{Q \cdot K^T}{\sqrt{d_k}})\cdot V^T$

  * Once all single-head attentions output their resultant matrices, they will all be concatenated, and the final concatenated matrix is once again transformed linearly by multiplying it with a set of weights matrix initialized with random values, which will later get updated when the transformer starts training.

  ![image-20240802170203371](./img/image-20240802170203371.png)

* Adding and Normalizing: Once we obtain the resultant matrix from multi-head attention, we have to add it to our original matrix. Then normalize the matrix.

  * To normalize the above matrix, we need to compute the mean and standard deviation row-wise for each row. We subtract each value of the matrix by the corresponding row mean and divide it by the corresponding standard deviation. Adding a small value of error prevents the denominator from being zero and avoids making the entire term infinity.

* After normalizing the matrix, it will be processed through a feedforward network.

* Adding and Normalizing Again: Once we obtain the resultant matrix from feed forward network, we have to add it to the matrix that is obtained from previous add and norm step, and then normalizing it using the row wise mean and standard deviation.

  * The output matrix of this add and norm step will serve as the query and key matrix in one of the multi-head attention mechanisms present in the decoder part.

----

* Decoder: We won’t be calculating the entire decoder because most of its portion contains similar calculations to what we have already done in the encoder. Instead, we only need to focus on the calculations of the input and output of the decoder.

  * When training, there are two inputs to the decoder. One is from the encoder, where the output matrix of the last add and norm layer serves as the **query** and **key** for the second multi-head attention layer in the decoder part. While the value matrix comes from the decoder after the first **add and norm** step.
  * The second input to the decoder is the predicted text. But the predicted input text needs to follow a standard wrapping of tokens that make the transformer aware of where to start and where to end.
    * `<start>` and `<end>` are two new tokens being introduced. 
    * The decoder takes one token as an input at a time. It means that `<start>` will be served as an input.

* Understanding Mask Multi Head Attention: In a Transformer, the masked multi-head attention is like a spotlight that a model uses to focus on different parts of a sentence. It’s special because it doesn’t let the model cheat by looking at words that come later in the sentence. Now, let’s understand the masked multi-head attention components having two heads:

  1. **Linear Projections (Query, Key, Value)**: Assume the linear projections for each head: Head 1: Wq1,Wk1,Wv1 and Head 2: Wq2,Wk2,Wv2

  2. **Calculate Attention Scores:** For each head, calculate attention scores using the dot product of Query and Key, and apply the mask to prevent attending to future positions.

  3. **Apply Softmax:** Apply the softmax function to obtain attention weights.

  4. **Weighted Summation (Value):** Multiply the attention weights by the Value to get the weighted sum for each head.

  5. **Concatenate and Linear Transformation:** Concatenate the outputs from both heads and apply a linear transformation.

     > This step helps capture different aspects of the input data from multiple perspectives, contributing to a richer representation that the model can use for further processing.

* Calculating the Predicted Word: 
  * The output matrix of the last add and norm block of the decoder must contain the same number of rows as the input matrix.
  * The last **add and norm block** resultant matrix of the decoder must be flattened in order to match it with a linear layer to find the predicted probability of each unique word in our dataset (corpus). This flattened layer will be passed through a linear layer to compute the **logits** (scores) of each unique word in our dataset.
  * Once we obtain the logits, we can use the **softmax** function to normalize them and find the word that contains the highest probability.
  * This predicted word (such as `you`), will be treated as the input word for the decoder, and this process continues until the `<end>` token is predicted.





[^1]: [Solving Transformer by Hand: A Step-by-Step Math Example | by Fareed Khan | Level Up Coding (gitconnected.com)](https://levelup.gitconnected.com/understanding-transformers-from-start-to-end-a-step-by-step-math-example-16d4e64e6eb1)
[^2]: [[1706.03762\] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)

