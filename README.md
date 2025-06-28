
# LLM Assignments Portfolio
### This repository contains all the assignments assigned in LLM course KU AICL434
This portfolio presents a collection of assignments exploring various facets of Natural Language Processing (NLP), Large Language Models (LLMs), and Multimodal AI. From foundational text preprocessing to advanced transformer fine-tuning, multi-agent systems, and image captioning, these projects highlight practical implementations and key concepts in modern AI.

---

## Assignment 1: Core NLP Techniques & Word Embeddings

This assignment provided foundational experience with preparing textual data and transforming words into numerical representations for machine learning.

### 1.1 NLP Text Preprocessing Techniques

This section focused on essential methods for cleaning and standardizing raw text.

-   **Tokenization**: Breaking text into smaller units (words or sentences) using libraries like NLTK and spaCy.
-   **Lemmatization**: Reducing words to their base or dictionary form, considering context (e.g., "running" to "run").
-   **Stemming**: A more aggressive method to reduce words to their root by chopping suffixes (e.g., "beautiful" to "beauti").
-   **Part-of-Speech (POS) Tagging**: Identifying the grammatical category of each word (e.g., noun, verb).
-   **Named Entity Recognition (NER)**: Classifying named entities (persons, organizations, locations) within text.

**Lemmatization vs. Stemming Comparison:**
-   **Linguistic Accuracy**: **Lemmatization** is more accurate, providing valid words; **stemming** can result in non-words.
-   **Irregular Forms**: **Lemmatization** handles irregular forms well (e.g., "better" to "good"); **stemming** often struggles.
-   **Speed vs. Accuracy**: **Stemming** is faster but less precise; **lemmatization** is more computationally intensive but offers higher linguistic precision.

### 1.2 Word Embeddings and Visualization

This part explored techniques to convert words into numerical vectors and visualize their relationships.

-   **Custom Corpus**: A small, categorized corpus was created to facilitate practical demonstrations.
-   **Embedding Techniques**:
    -   **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure implemented with `scikit-learn` to quantify word relevance in documents. TF-IDF typically generates sparse, high-dimensional vectors and excels in document retrieval.
    -   **GloVe (Global Vectors for Word Representation)**: An unsupervised algorithm leveraging global co-occurrence statistics. Pre-trained GloVe embeddings were used to capture semantic relationships, resulting in dense, lower-dimensional vectors suitable for semantic tasks.
-   **Visualization**: Dimensionality reduction techniques were employed to visualize embeddings:
    -   **PCA (Principal Component Analysis)**: A linear method for reducing feature dimensions.
    -   **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear method ideal for visualizing high-dimensional data in 2D or 3D, revealing clusters.

**TF-IDF vs. GloVe Comparison:**
-   **Principle**: **TF-IDF** relies on term frequency; **GloVe** on global word co-occurrence counts.
-   **Semantic Understanding**: **TF-IDF** lacks inherent semantic understanding; **GloVe** captures word similarities.
-   **Structure**: **TF-IDF** produces sparse vectors; **GloVe** produces dense vectors.

---

## Assignment 2: Text Summarization

This assignment focused on developing a model to generate concise summaries (headlines) from longer news articles.

-   **Data**: The `news_summary.csv` dataset was used, containing `headlines` and `text` columns. Data integrity was ensured by removing duplicates.
-   **Data Preprocessing**: A comprehensive pipeline was applied:
    -   Lowercase conversion, contraction expansion, stop word removal.
    -   Handling of special characters and consistent spacing.
    -   Addition of `_START_` and `_END_` tokens for sequence boundary marking.
-   **Model Architecture**: A **Sequence-to-Sequence (Seq2Seq)** neural network, built with Keras/TensorFlow, was employed. It typically features an encoder-decoder framework with **Embedding**, **LSTM** (including Bidirectional), **Concatenate**, **TimeDistributed**, and **Dense** layers, often enhanced with an attention mechanism.
-   **Training**: An **EarlyStopping** callback was implemented to prevent overfitting and optimize training time by monitoring validation metrics.
-   **Evaluation**: Model performance was assessed using **cosine similarity** between generated and actual summaries. The model achieved a **mean cosine similarity score of approximately 0.424**, indicating reasonable textual similarity.

---

## Assignment 3: Transformer Fine-tuning for Text Classification

This assignment demonstrated the power of **transfer learning** by fine-tuning a pre-trained transformer model for a text classification task.

-   **Dataset**: A text classification dataset, likely similar to AG News or IMDb reviews, consisting of news articles categorized into various classes.
-   **Model**: **DistilBERT**, a smaller and more efficient version of BERT, was used. Fine-tuning involves adapting a model pre-trained on a vast text corpus to a specific, smaller dataset.
-   **Preprocessing & Tokenization**: Text data was processed using a DistilBERT-appropriate tokenizer, involving tokenization, handling special tokens, padding, truncation, and encoding.
-   **Training**: The model was trained with specified epochs and batch sizes. **Callbacks** like **EarlyStopping** and potentially **ModelCheckpoint** were used for monitoring validation metrics and saving optimal models. An **optimizer** (e.g., AdamW) updated model weights.
-   **Evaluation**: Standard classification metrics were used:
    -   **Accuracy**: Overall correct classifications.
    -   **F1-Score (Macro)**: Unweighted mean F1, treating all classes equally.
    -   **F1-Score (Weighted)**: F1 score weighted by class support, considering imbalance.
    -   Key results included **Test Accuracy**, **Test F1-Scores**, and the **Best Validation Accuracy**.

---

## Assignment 4: Prompt Engineering Techniques

This assignment explored various strategies for designing and refining prompts to optimize Large Language Model (LLM) outputs, focusing on accuracy, clarity, and reasoning.

### 4.1 Prompt Design Techniques

Explored different prompt types using a question-answering task (e.g., about the Eiffel Tower).

-   **Direct Prompt**: Simple question, quick answer. Lacks reasoning transparency.
-   **Few-Shot Prompt**: Provides input-output examples. Improves consistency and format adherence.
-   **Chain-of-Thought (CoT) Prompt**: Instructs the model to break down reasoning steps. Significantly enhances accuracy, clarity, and provides insight into the model's logic.
    -   **Conclusion**: CoT offers the best balance of accuracy, clarity, and explainability for complex tasks.

### 4.2 Prompt Tuning Techniques

Focused on refining prompts for sentiment analysis to improve accuracy, clarity, and consistency.

-   **Base Prompt (Baseline)**: Initial, straightforward instruction. May lack specificity.
-   **Manual Refinement**: Direct modification for clarity and enforcing output format (e.g., "Answer with 'Positive' or 'Negative'.").
-   **Chain-of-Thought (CoT) for Sentiment Analysis**: Guides the model through logical steps (e.g., "Identify emotional tone," "Look for keywords"). Leads to highly accurate results with transparent reasoning, especially for nuanced sentiment.

**Overall Conclusion for Prompt Engineering**: Strategic prompt design and iterative tuning are critical for maximizing LLM effectiveness. **Few-Shot learning** enhances consistency, **Manual Refinement** improves clarity, and **Chain-of-Thought (CoT)** is a powerful technique across tasks for better accuracy, explainability, and handling complex scenarios.

---

## Assignment 5: Multimodal Image Captioning

This assignment implemented a **multimodal image captioning system**, demonstrating how a transformer-based model can generate descriptive text from images.

-   **Model Used**: The core system utilizes the **BLIP (Bootstrapping Language-Image Pre-training)** model (`BlipProcessor` and `BlipForConditionalGeneration` from Hugging Face Transformers). BLIP is effective at integrating visual and textual modalities.
-   **Process Pipeline**:
    1.  **Image Loading**: Images are loaded from local files (using PIL) or URLs (using `requests`).
    2.  **Preprocessing (Processor)**: The `BlipProcessor` prepares the image by handling transformations and tokenization, converting it into a numerical format for the model.
    3.  **Caption Generation**: The preprocessed image is fed into `BlipForConditionalGeneration` to produce a textual description, which is then decoded back into a human-readable string.
-   **Example Scenario**: A practical demonstration involved generating a caption for a local image file.
    -   **Input Image**: `grasshopper.jpeg` (an image of a coffee cup).
    -   **Generated Caption**: "a large coffee cup" (accurately describing the coffee cup within the grasshopper-named file).