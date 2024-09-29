import pickle
import pandas as pd
medicines_dict = pickle.load(open('medicine_dict.pkl','rb'))
medicines = pd.DataFrame(medicines_dict)

                    # To load similarity-vector-data from pickle in the form of dictionary
similarity = pickle.load(open('similarity.pkl','rb'))
def recommend(medicine):
     medicine_index = medicines[medicines['Drug_Name'] == medicine].index[0]
     distances = similarity[medicine_index]
     medicines_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

     recommended_medicines = []
     for i in medicines_list:
         recommended_medicines.append(medicines.iloc[i[0]].Drug_Name)
     return recommended_medicines
