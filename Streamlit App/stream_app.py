import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image

model_file = 'mejor_modelo.pkl'

model = pickle.load(open(model_file, 'rb'))
X_scaler_train = pd.read_csv('X_scaler_parameters.csv')

def main():

	# image = Image.open('images/icone.png')
	image2 = Image.open('images/image.png')
	# st.image(image,use_column_width=False)
	add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?",
	("Online", "Batch"))
	st.sidebar.info('This app is created to predict Customer Churn')
	st.sidebar.image(image2)
	st.title("Customer Churn Prediction - TELCO")
	if add_selectbox == 'Online':
		Satisfaction_Score = st.number_input(' Satisfaction Score :' , min_value=1, max_value=5, value=3)
		Online_Security = st.selectbox(' Online Security: ', ['Yes', 'No'])
		Contract = st.selectbox(' Contrato : ', ['Month-to-Month', 'One Year', 'Two Year'])
		Senior_Citizen = st.selectbox(' Senior Citizen : ', ['Yes', 'No'])
		Number_of_Referrals = st.number_input(' Number of Referrals: ' , min_value=0, max_value=11, value=0)
		Dependents = st.selectbox(' Dependents: ', ['Yes', 'No'])
		Tenure_in_Months = st.number_input(' Tenure in Months :' , min_value=1, max_value=100, value=2)
		Internet_Service = st.selectbox(' Customer has Internet Service:', ['Yes', 'No'])
		Paperless_Billing= st.selectbox(' Customer has a Paperless Billing:', ['Yes', 'No'])
		Total_Extra_Data_Charges = st.number_input(' Total Extra Data Charges :', min_value=1, max_value=100, value=1)
		paymentmethod= st.selectbox('Payment Method:', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])
		Referred_a_Friend = st.selectbox(' Referred a Friend : ', ['Yes', 'No'])
		Streaming_Music = st.selectbox(' Streaming Music : ', ['Yes', 'No'])
		Streaming_Movies = st.selectbox(' Streaming Movies : ', ['Yes', 'No'])    
		Monthly_Charge = st.slider(' Monthly Charge : ', 0, 120, 25)
		Total_Revenue = st.slider(' Total Revenue : ', 0, 20000, 1000)
		Device_Protection_Plan = st.selectbox(' Device Protection Plan : ', ['Yes', 'No']) 
		output= ""
		output_prob = ""
  
		input_dict={
				"Satisfaction Score":Satisfaction_Score,
				"Online Security": Online_Security,
				"Contract": Contract,
				"Senior Citizen": Senior_Citizen,
				"Number of Referrals": Number_of_Referrals,
				"Dependents": Dependents,
				"Tenure in Months": Tenure_in_Months,
				"Internet Service": Internet_Service,
				"Paperless Billing": Paperless_Billing,
				"Total Extra Data Charges": Total_Extra_Data_Charges,
				"Payment Method": paymentmethod,
				"Referred a Friend": Referred_a_Friend,
				"Streaming Music": Streaming_Music,
				"Streaming Movies": Streaming_Movies,
				"Monthly Charge": Monthly_Charge,
				"Total Revenue": Total_Revenue,
				"Device Protection Plan": Device_Protection_Plan,
			}

		if st.button("Predict"):
			
			X = pd.DataFrame.from_dict(input_dict, orient='index').T
   
			for i in ['Online Security','Senior Citizen','Dependents','Internet Service','Paperless Billing','Referred a Friend','Streaming Music','Streaming Movies',
             			'Device Protection Plan']:
					X[i] = X[i].map({'Yes':1,'No':0})
     
			X['Contract'] = X['Contract'].map({'Month-to-Month':0,'One Year':1, 'Two Year':2 })
			X['Payment Method'] = X['Payment Method'].map({'Bank Withdrawal':0,'Credit Card':1, 'Mailed Check':2 })

			scaler = StandardScaler()
			scaler.fit(X_scaler_train)
   
			X = scaler.transform(X)
   
			y_pred = model.predict_proba(X)[0, 1]
			churn = model.predict(X)[0] == 1
			output_prob = float(y_pred)
			output = bool(churn)
		st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))
  
	if add_selectbox == 'Batch':
		file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
		if file_upload is not None:
      
			data = pd.read_csv(file_upload)
   
			X = data
   
			for i in ['Online Security','Senior Citizen','Dependents','Internet Service','Paperless Billing','Referred a Friend','Streaming Music','Streaming Movies',
             			'Device Protection Plan']:
					X[i] = X[i].map({'Yes':1,'No':0})
     
			X['Contract'] = X['Contract'].map({'Month-to-Month':0,'One Year':1, 'Two Year':2 })
			X['Payment Method'] = X['Payment Method'].map({'Bank Withdrawal':0,'Credit Card':1, 'Mailed Check':2 })

			scaler = StandardScaler()
			scaler = StandardScaler()
			scaler.fit(X_scaler_train)
   
			X = scaler.transform(X)
   
			y_pred = model.predict_proba(X)[0, 1]
			churn = model.predict(X)[0] == 1
			churn = bool(churn)
			st.write(churn)

if __name__ == '__main__':
	main()