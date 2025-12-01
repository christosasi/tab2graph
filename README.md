A schema to Graph Extraction package for tabular datasets


   ```python
   def extract_graph(node_columns, edge_columns, feature_spec):

       def extract_nodes(node_columns):
           ...
           return node_index_tensor
       def extract_edges(edge_columns):
           ...
           return edge_index_tensor

       '''
       Numpy, scklearn, math or torch packages wherever available 
       Best Open Source candidate is feature engine repo.
       def one_hot_encoding
       def multi_hot
       def z_score_norm
       def boolean_cast
       def dense_projection
       def embedding_bag
       def time_delta_norm
       '''
       def extract_features(feature_spec):
           feature_map = encoding_agent(feature_spec) 
           for column_names, features in feature_map:
               ...
               call_encoding_functions
               ... 
               append feature_tensor
              
           return feature_tensor

        #LLM may adjust this dynamically to generate the final tensor with correct shape
        final_graph_tensor = concat(node_tensor, edge_tensor, feature_matrix_tensor)
        return final_graph_tensor

   graph = extract_graph(
       node_columns=["drug_id", "gene_id", "disease_id", "trial_id"],
       edge_columns=[("drug_id", "treats", "disease_id"),
                     ("drug_id", "targets", "gene_id"),
                     ("trial_id", "tests", "drug_id")],
       feature_spec={
           "drug_id": ["atc_code", "structure_fingerprint"],
           "gene_id": ["expression_level", "pathway_membership"],
           "disease_id": ["phenotype_score"],
           "trial_id": ["phase", "n_patients", "endpoint_success"]
       }
   )
   
    #LLM needs to be called after feature_spec definition to assign appropriate feature encoding
   feature_encoding = {
        "categorical": "one_hot",
        "multi_label": "multi_hot",
        "numeric": "z_score_normalization",
        "ordinal": "integer_encoding",
        "binary": "boolean_cast",
        "high_dim_fingerprints": "dense_projection",
        "text_features": "embedding_bag",
        "temporal_features": "time_delta_norm"
    }

    
    
