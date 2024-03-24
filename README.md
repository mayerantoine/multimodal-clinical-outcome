# multimodal-clinical-outcome
Reproducing the paper Improving Clinical Outcome Predictions Using Convolution over Medical Entities with Multimodal Learning


### How to use for now

* run time series pipeline
* run etl to extract clinical notes
* run the code to extract clinical entities using med7
* train word2vec model on the entities
* create patients embeddigns from word2vec and entities extracted
* Run training model

```
    python timeseries_features_pipeline.py
    python etl_clinical_notes.py
    python extract_clinical_entities.py
    python train_embeddings.py
    python feature_embeddings.py
    python train_proposed_model.py
    
```
