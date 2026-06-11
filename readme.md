egal Contract RAG Chatbot
Project Description
This project is a chatbot that can answer questions about legal contracts using Retrieval-Augmented Generation (RAG).
The chatbot uses the CUAD (Contract Understanding Atticus Dataset), which contains real commercial contracts and labeled contract clauses. Instead of relying only on the language model, the system searches the contract for relevant information and uses that information to generate answers.
For example, a user can ask:
What are the termination conditions?
Is there a confidentiality clause?
What happens if a party breaches the agreement?
The system retrieves the most relevant sections of the contract and provides an answer based on those sections.
Dataset
CUAD (Contract Understanding Atticus Dataset)
https://zenodo.org/records/4595826
Current Plan
Load and preprocess contract data
Split contracts into smaller chunks
Generate embeddings for each chunk
Store embeddings in a vector database
Retrieve relevant chunks based on user questions
Generate answers using an LLM
Goal
The goal of this project is to build and evaluate a RAG-based chatbot for legal contract analysis and compare different retrieval approaches to improve answer accuracy.
