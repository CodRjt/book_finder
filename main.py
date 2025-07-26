import streamlit as st 
import pandas as pd
from sentence_transformers import SentenceTransformer,util
from torch import topk
st.title("Find me Book")

st.write("## Readaa !!!")

search=st.text_input("Enter a book or author you like")
# Author_name=st.text_input("Enter an author you like")
# genre_name=st.text_input("Enter a genre you like")

# def fuzzy_finder(df:pd.DataFrame,query,columns):
#     search_pool=df[columns].astype(str).agg(" ".join,axis=1)
#     # st.write(search_pool.tolist())
#     result=process.extract(query,search_pool.tolist(),limit=None)
#     data=[df.iloc[match[2]] for match in result if match[1] > 50]
#     return pd.DataFrame(data)

model=SentenceTransformer('all-MiniLM-L6-v2')
def semantic_finder(df:pd.DataFrame,query,columns):
    search_pool=df[columns].astype(str).agg(" ".join,axis=1)
    # st.write(search_pool.tolist())
    search_pool_embeddings=model.encode(search_pool.tolist(),convert_to_tensor=True)
    query_embedding=model.encode(query,convert_to_tensor=True)

    cosine_scores=util.pytorch_cos_sim(query_embedding,search_pool_embeddings)
    return topk(cosine_scores,k=8)

    
data=pd.read_csv("books.csv")
if search:
    filtered=semantic_finder(data,search,["authors","title"])
    # st.write(filtered)
    st.write("Top 8 matching books")
    # st.write(f"found {len(filtered)} matching books")
    # for _,row in filtered.head(10).iterrows():
    for idx,score in zip(filtered.indices[0],filtered.values[0]):
        row=data.iloc[idx.item()]
        col1,col2=st.columns(2)
        with col1:
            st.image(row['image_url'])
        with col2:
            st.subheader(row['original_title'])
            st.write(f"Author:{row['authors']}")
            st.write(f"Year:{row['original_publication_year']}")
            st.write(f"Language: {row['language_code']}")
            st.write(f"Ratings:{row['average_rating']}({row['ratings_count']}) rating")
        

