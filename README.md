Mart - Contract Vulnerability Detection using MLM and CL
This project implements a smart contract vulnerability detection framework based on Contrastive Learning Enhanced Automated Recognition (Clear). The method leverages both Masked Language Modeling (MLM) and Contrastive Learning (CL) to capture fine-grained correlation information among smart contracts. This, in turn, improves the detection of subtle vulnerabilities that might be overlooked when treating each contract as an isolated entity.

Overview
Smart contracts are critical components in blockchain systems, yet they are prone to vulnerabilities that can lead to significant financial losses. This project aims to improve vulnerability detection by:

Sampling Contract Pairs: Generating pairs of contracts to capture inter-contract relationships.

Contrastive Learning: Using a dual loss function that combines MLM and contrastive loss to learn detailed semantic and structural representations.

Vulnerability Detection: Fine-tuning the learned representations to accurately classify contracts as vulnerable or non-vulnerable.

Process
1. Data Sampling
Objective: Establish correlations between smart contracts.

Method:

Vulnerable Contract Extraction: Identify all vulnerable contracts from the dataset.

Pairing Strategy: For each contract, randomly select another contract from the vulnerable set to form a pair.

Correlation Labeling:

Assign a label of 1 if both contracts in the pair are vulnerable (V-V).

Assign a label of 0 if one contract is vulnerable and the other is not (V-N).

2. Contrastive Learning Module
Contextual Augmentation:

Randomly mask 30% of the tokens in the contract code.

Use a Transformer-based Masked Language Model (MLM) to predict the masked tokens. This step helps the model learn the contextual semantics and structure of the contract code.

Feature Learning:

Extract a global representation using the special CLS token from the Transformer output.

Integrate positional encoding with token-level features via multi-head attention to capture fine-grained information.

Contrastive Loss Computation:

Compute the Euclidean distance between the global vector representations of contract pairs.

Optimize the model with a combination of contrastive loss (which reinforces similarity for V-V pairs and dissimilarity for V-N pairs) and MLM loss.

3. Vulnerability Detection Stage
Fine-Tuning:

The Transformer model is fine-tuned using the feature representations from the CL module.

Both the global semantic vector and token-level features are concatenated.

Classification:

A fully connected neural network predicts whether a smart contract is vulnerable.

A sigmoid activation function is applied to produce a probability, and the prediction is compared against the true label using a cross-entropy loss function.
