import pandas as pd
from src.models.bert_model import BERTEmbedder
from src.utils.utils import load_marr_compl_df

def generate_embeddings():
    try:
        # Initialize BERT embedder
        print("Initializing BERT embedder...")
        embedder = BERTEmbedder()
        
        # Load marr_new_df
        print("Loading elmr_compl_df...")
        marr_new_df = load_marr_compl_df()

        # Generate embeddings for marr_new_df
        print("Generating embeddings for marr_new_df...")
        embeddings = embedder.embed_dataframe(marr_new_df, 'text')
        print(f"Generated embeddings of shape: {embeddings.shape}")
        
        # Create a copy of the original DataFrame
        df_with_embeddings = marr_new_df.copy()
        
        # Add embeddings as columns
        for i in range(embeddings.shape[1]):
            df_with_embeddings[f'embedding_{i}'] = embeddings[:, i]
        
        # Save to CSV
        output_file = 'notebooks/marr_new_with_embeddings.csv'
        df_with_embeddings.to_csv(output_file, index=False)
        print(f"Saved DataFrame with embeddings to {output_file}")
        print(f"Total columns: {len(df_with_embeddings.columns)}")
        print(f"Embedding columns: {len([col for col in df_with_embeddings.columns if col.startswith('embedding_')])}")
        
        return df_with_embeddings
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    generate_embeddings()
