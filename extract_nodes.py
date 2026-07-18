#Extract_node function
import logging

def extract_nodes([node_columns]):
   """Function should handle unique IDs from all Columns passed in the column_list
      Data types such as strings and integers should be handled  

   """
   node_tensor = empty_tensor
   for col in node_columns:
       node_tensor = node_tensor.append(unique(col)) # Could this be more efficient?
       INFO.logging=(f" %len(node_tensor) added from %col") 
    # Do we need to explicity right an if-else here for the append so that deduplication does not take place?
    # Should we first build a set of all collected UIDs and then finally generate a tensor?
   
   INFO.logging=(f" %len(node_tensor) Nodes Extracted:") 
   return node_index_tensor

