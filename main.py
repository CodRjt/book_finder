import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import math

st.title("Find me Book")

st.write("## Your Interest")

search=st.text_input("Enter a book or author you like")
# Author_name=st.text_input("Enter an author you like")
# genre_name=st.text_input("Enter a genre you like")

data=pd.read_csv("books.csv")
if search:
    filtered=data[data['original_title'].str.contains(search,case=False)| data['authors'].str.contains(search,case=False)]
    st.write(f"found {len(filtered)} matching books")
    for _,row in filtered.head(10).iterrows():
        col1,col2=st.columns(2)
        with col1:
            st.image(row['image_url'])
        with col2:
            st.subheader(row['original_title'])
            st.write(f"Author:{row['authors']}")
            st.write(f"ratings:{row['average_rating']}")
        

