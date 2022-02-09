### About

This repository contains materials for the NAACL 2021 paper [Modeling Framing in Immigration Discourse on Social Media](https://aclanthology.org/2021.naacl-main.179.pdf). 

- ``dataset.zip`` contains the full set of tweet IDs used for analysis. Human-annotated data for training frame detection models is located in the ``annotated_data``
folder, and machine-predicted frame labels are located in the ``predicted_data`` folder. 

- ``codebook.pdf`` contains guidelines for frame annotation. It includes detailed descriptions of issue-generic policy, immigration-specific, and episodic/thematic frames. 

- ``code/`` contains all code for data collection, assessing annotations, and building and evaluating models
- ``notebooks/`` contain Jupyter notebooks for framing analyses, including regressions and plots

### Frame Detection Models
Multilabel RoBERTa classification models for identifying frames, fine-tuned on our full set of immigration-related tweets, can be found here:  

- https://huggingface.co/juliamendelsohn/framing_issue_generic 
- https://huggingface.co/juliamendelsohn/framing_immigration_specific
- https://huggingface.co/juliamendelsohn/framing_narrative

Please see [this Colab notebook](https://colab.research.google.com/drive/1b_cqXwjqH-uk_N7OdtdRv1_5lS9o2oJH) for how to use the frame classification models. 

### Citation 
```
  @inproceedings{mendelsohn2021modeling,
  title={Modeling Framing in Immigration Discourse on Social Media},
  author={Mendelsohn, Julia and Budak, Ceren and Jurgens, David},
  booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={2219--2263},
  year={2021}
}
  ```
  ### Contact 
  
  Please email Julia Mendelsohn (juliame@umich.edu) with any issues, questions, or comments.
