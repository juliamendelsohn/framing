### About

This repository contains materials for the NAACL 2021 paper [Modeling Framing in Immigration Discourse on Social Media](https://arxiv.org/abs/2104.06443). 

- ``dataset.zip`` contains the full set of tweet IDs used for analysis. Human-annotated data for training frame detection models is located in the ``annotated_data``
folder, and machine-predicted frame labels are located in the ``predicted_data`` folder. 

- ``codebook.pdf`` contains guidelines for frame annotation. It includes detailed descriptions of issue-generic policy, immigration-specific, and narrative frames. 

- ``code/`` contains all code for data collection, assessing annotations, and building and evaluating models
- ``notebooks/`` contain Jupyter notebooks for framing analyses, including regressions and plots

### Frame Detection Models
Multilabel RoBERTa classification models for identifying frames, fine-tuned on our full set of immigration-related tweets, can be found here:  

- https://huggingface.co/juliamendelsohn/framing_issue_generic 
- https://huggingface.co/juliamendelsohn/framing_immigration_specific
- https://huggingface.co/juliamendelsohn/framing_narrative

### Citation 
```
@inproceedings{mendelsohn2021,
      title={Modeling Framing in Immigration Discourse on Social Media}, 
      author={Julia Mendelsohn and Ceren Budak and David Jurgens},
      year={2021},
      arxivId={2104.06443},
      archivePrefix={arXiv},
      booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      publisher = "Association for Computational Linguistics",
  }
  ```
  ### Contact 
  
  Please email Julia Mendelsohn (juliame@umich.edu) with any issues, questions, or comments.
