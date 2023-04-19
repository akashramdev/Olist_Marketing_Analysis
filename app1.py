import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
#######################################################
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")


#######################################################
df_ouptput1 = pd.read_excel("C:/Users/acer/Downloads/SpJain/Dubai/Projects/sem 4/AIM/output1.xlsx")

user_input = st.text_input("Enter Customer ID")

df_ouptput1.loc[df_ouptput1['customer_id']==user_input].iloc[:,-4:]

#######################################################

adamodel = pickle.load(open('C:/Users/acer/Downloads/SpJain/Dubai/Projects/sem 4/AIM/Code/adamodel.pkl', 'rb'))

uploaded_file = st.file_uploader("Choose an excel file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the file into a pandas DataFrame
    data = pd.read_excel(uploaded_file)

    data1 = data.copy()

    # Run the model on the data
    predictions = adamodel.predict(data)

    data1['Segment'] = predictions

    data1['Segment'].replace({0:'New Customer', 1:'Churn customers', 2:'Need Attention',  3: 'Potential Loyal Customers', 4: 'Loyal Customers'},inplace=True)
    
    # Display the predictions
    st.write("Predictions:", data1)

#######################################################
    # Product Recommendation 

    rec1 = pd.read_excel("C:/Users/acer/Downloads/SpJain/Dubai/Projects/sem 4/AIM/recommendation.xlsx")
    
    #rec1.loc

    
   

    order_rating = rec1.loc[:, ['customer_id', 'product_id', 'review_score']]

    order_rating = order_rating[order_rating['product_id'].isin(
        order_rating['product_id'].value_counts()[
            order_rating['product_id'].value_counts() > 10].index)]
    
    order_rating = order_rating.reset_index()

    ratings_utility_matrix = order_rating.pivot_table(values='review_score',
                                                  index='customer_id',
                                                  columns='product_id',
                                                  fill_value=0)
    #ratings_utility_matrix.head()

    X = ratings_utility_matrix.T


    def fitsystemrecommendation(ratings_utility_matrix):
        X = ratings_utility_matrix.T
        SVD = TruncatedSVD(n_components=10)
        decomposed_matrix = SVD.fit_transform(X)
        correlation_matrix = np.corrcoef(decomposed_matrix)
        return correlation_matrix
    
    correlation_matrix = fitsystemrecommendation(ratings_utility_matrix)

    def systemrecommendation(prod_id):
        order_rating.index[order_rating['product_id'] == prod_id].tolist()[1]
        product_names = list(X.index)
        product_ID = product_names.index(prod_id)
        correlation_product_ID = correlation_matrix[product_ID]
        Recommend = list(X.index[correlation_product_ID > 0.70])
        Recommend.remove(prod_id)
        op = pd.DataFrame(Recommend[0:9], columns=['Recommendation'])
        return st.write("Recommendation",op)
    
    user_input1 = st.text_input("Enter Product ID")
    #systemrecommendation(user_input1)
    
    if st.button('Recommend'):
        systemrecommendation(user_input1)