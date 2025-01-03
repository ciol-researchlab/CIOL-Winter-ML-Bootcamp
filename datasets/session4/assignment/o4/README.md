About Dataset
This is a dataset for the sentiment analysis task. It contains 100k code mixed data. The languages are Bangla-English-Hindi.

Dataset Generation:
Initially, we select the Amazon Review Dataset as our base data, referenced from Ni et al. (2019)[1]. We randomly extract 100,000 instances from this dataset. The original labels in this dataset are ratings, scaled from 1 to 5. For our specific task, we categorize them into Positive (rating > 3), Neutral (rating = 3), and Negative (rating < 3), ensuring a balanced number of instances for each label. To generate the synthetic Code-mixed dataset, we apply two distinct methodologies: the Random Code-mixing Algorithm by Krishnan et al. (2021)[2] and r-CM by Santy et al. (2021)[3].



@article{raihan2023mixed,
title={Mixed-Distil-BERT: Code-mixed Language Modeling for Bangla, English, and Hindi},
author={Raihan, Md Nishat and Goswami, Dhiman and Mahmud, Antara},
journal={arXiv preprint arXiv:2309.10272},
year={2023}
}

References
[1]: Ni, J., Li, J., & McAuley, J. (2019). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP) (pp. 188-197).

[2]: Krishnan, J., Anastasopoulos, A., Purohit, H., & Rangwala, H. (2021). Multilingual code-switching for zero-shot cross-lingual intent prediction and slot filling. arXiv preprint arXiv:2103.07792.

[3]: Santy, S., Srinivasan, A., & Choudhury, M. (2021). BERTologiCoMix: How does code-mixing interact with multilingual BERT? In Proceedings of the Second Workshop on Domain Adaptation for NLP (pp. 111-121).