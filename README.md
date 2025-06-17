# Financial News Sentiment Classification with Deep Learning  

**Course:** Deep Learning Applications 

> “Markets are moved by stories as much as by spreadsheets.”

---

## Project at a Glance  
* **Task:** Three-class, sentence-level **sentiment classification** of financial headlines (*negative · neutral · positive*).  
* **Learning Paradigm:** **Supervised** NLP with both traditional models and modern **deep-learning algorithms**.  
* **Why it matters:** Traders and risk desks face headline overload; an accurate, explainable classifier surfaces the few lines that truly move prices.

---

## Dataset  
| Item | Value |
|------|-------|
| Source | **Financial Sentiment Analysis** (Bhatti 2023) – FiQA v1.1 + Financial PhraseBank |
| Size | **5 842 sentences / 680 KB** |
| Avg length | 117 chars (~38 WordPiece tokens) |
| Label mix | Neutral 54 % · Positive 32 % · Negative 14 % |
| Integrity | 0 nulls, 37 duplicates removed → 5 805 unique rows |
| License | CC-0 1.0 (public domain) |

Annotations come from domain experts and business-school students (κ ≈ 0.75).

---

## Data Cleaning Highlights  
* Lower-case, strip URLs, but **keep `$` tickers** and stop-words for negation cues.  
* Drop exact duplicates to avoid train/test leakage.  
* No stemming/lemmatisation to stay aligned with BERT’s sub-word vocabulary.  
* Class imbalance handled later via **class-weighted loss** (+ optional oversampling for RNNs).  

---

## Exploratory Data Analysis (EDA)  
* **Imbalance:** Neutral ≈ 3.7 × Negative → macro-F1 chosen as main metric.  
* **Length:** 95 % of sentences < 200 chars → pad/truncate to 50 tokens.  
* **Correlation heat-map** of top TF-IDF bigrams shows strong pairings (“beat | estimates” = Positive).  
* **χ² tests** confirm “skepticism”, “lawsuit” as highly discriminatory for Negative.  
* **Temporal drift check:** 2008–09 spike in negatives (crisis), but post-2010 ratios stable enough for random splits.

---

## Models Evaluated  
| Model | Key Idea | Params | Macro-F1 |
|-------|----------|-------:|---------:|
| TF-IDF + LogReg | Sparse baseline | 50 k | 0.68 |
| Bi-LSTM | Sequential context | 2 M | 0.66 |
| Bi-LSTM + Attention | Focus on salient tokens | 2 M | 0.61* |
| **FinBERT** | Finance-pre-trained transformer | 109 M | **0.71** |

\* Mask-aware attention layer needed; current toy layer drops performance but provides interpretability.

---

## Key Results  
* **FinBERT** leads macro-F1 (+ 0.03 vs baseline) and overall accuracy (0.80).  
* **Bi-LSTM** attains best Negative recall (0.66) ­– valuable when missing bearish news is costly.  
* Class weighting + oversampling lifted Negative recall by **+0.19** overall.  
* Attention heat-maps highlight “skepticism”, “lawsuit”, etc., offering token-level rationale for compliance.

---

## Lessons Learned  
1. **Domain pre-training trumps architecture size**: FinBERT > bigger generic BERT in pilot tests.  
2. **Mask handling matters**: ignoring padding in attention restored lost F1 in offline trials.  
3. **Class weighting beats oversampling** for transformers; opposite for small RNNs.  

---

## Next Steps  
* Scale labels to 1 M headlines via **weak supervision** (Snorkel).  
* Try **focal loss** & threshold calibration to push Negative recall > 0.80.  
* Deploy as a **FastAPI** micro-service with Redis caching; Distil/quantise FinBERT for CPU latency < 10 ms.  
* Continuous nightly retraining to combat vocabulary drift (“AI boom”, “Web3”, etc.).

---

## References  
* Bhatti, S. (2023). *Financial Sentiment Analysis [Dataset].* Kaggle.  
* Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.*  
* Malo, P. et al. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts.*  
* Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*  


