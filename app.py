import numpy as np
import pandas as pd
import streamlit as st
import os
import joblib

def home():
    pass

def Prediction():
    
    model = joblib.load("catboost_model.pkl")

    st.title("Credit Card Prediction")

    st.subheader("Custmor Information")

    name = st.text_input("Enter Name: ")
    phone = st.number_input("Enter Phone number: ")
    email = st.text_input("Enter Email: ")

    limit_bal = st.number_input("Enter LIMIT_BAL: ", min_value= 0.0, value = 50000.0)

    sex_input = st.radio("Enter Sex: ", ["Male", "Female"])

    education_input = st.radio("Enter Education: ",["Graduate School","University","High School", "Others"])

    marriage_input = st.radio("Marriage: ", ["Married","Single", "Others"])

    age = st.number_input("Enter Age: ", min_value=18)



    st.write("-2: No consumption")
    st.write("-1: Paid duly")
    st.write("0: Revolving credit")
    st.write("1: Payment delay 1 month")
    st.write("2: Payment delay 2 months")
    st.write("3: Payment delay 3 months")
    st.write("4: Payment delay 4 months")
    st.write("5: Payment delay 5 months")
    st.write("6: Payment delay 6 months")
    st.write("7: Payment delay 7 months")
    st.write("8: Payment delay 8+ months")


    pay_0 = st.selectbox("Repayment Status (Pay_0): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_2 = st.selectbox("Repayment Status (Pay_2): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_3 = st.selectbox("Repayment Status (Pay_3): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_4 = st.selectbox("Repayment Status (Pay_4): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_5 = st.selectbox("Repayment Status (Pay_5): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    pay_6 = st.selectbox("Repayment Status (Pay_6): ", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])


    st.subheader("Bill Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:
        bill_amt1 = st.number_input("BILL_AMT1", value=0.0)
        bill_amt2 = st.number_input("BILL_AMT2", value=0.0)

    with col2:
        bill_amt3 = st.number_input("BILL_AMT3", value=0.0)
        bill_amt4 = st.number_input("BILL_AMT4", value=0.0)

    with col3:
        bill_amt5 = st.number_input("BILL_AMT5", value=0.0)
        bill_amt6 = st.number_input("BILL_AMT6", value=0.0)



    st.subheader("Payment Amounts")

    col1, col2, col3 = st.columns(3)

    with col1:
        pay_amt1 = st.number_input("PAY_AMT1", value=0.0)
        pay_amt2 = st.number_input("PAY_AMT2", value=0.0)

    with col2:
        pay_amt3 = st.number_input("PAY_AMT3", value=0.0)
        pay_amt4 = st.number_input("PAY_AMT4", value=0.0)

    with col3:
        pay_amt5 = st.number_input("PAY_AMT5", value=0.0)
        pay_amt6 = st.number_input("PAY_AMT6", value=0.0)



    sex_map = {"Male": 1, "Female": 2}
    edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    marriage_map = {"Married": 1, "Single": 2, "Others": 3}

    sex = sex_map[sex_input]
    education = edu_map[education_input]
    marriage = marriage_map[marriage_input]



    if "ID" not in st.session_state:
        st.session_state.ID = 0

    if "prediction" not in st.session_state:
        st.session_state.prediction = None


    if st.button("Predict"):

        st.session_state.ID += 1

        result = pd.DataFrame([{
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            "PAY_0": pay_0,
            "PAY_2": pay_2,
            "PAY_3": pay_3,
            "PAY_4": pay_4,
            "PAY_5": pay_5,
            "PAY_6": pay_6,
            "BILL_AMT1": bill_amt1,
            "BILL_AMT2": bill_amt2,
            "BILL_AMT3": bill_amt3,
            "BILL_AMT4": bill_amt4,
            "BILL_AMT5": bill_amt5,
            "BILL_AMT6": bill_amt6,
            "PAY_AMT1": pay_amt1,
            "PAY_AMT2": pay_amt2,
            "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4,
            "PAY_AMT5": pay_amt5,
            "PAY_AMT6": pay_amt6
        }])

        prediction = model.predict(result)[0]

        # Save prediction in session
        st.session_state.prediction = prediction

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Customer is likely to default.")
        else:
            st.success("Customer is unlikely to default.")

        probability = model.predict_proba(result)[0]

        st.write(f"Not Default: {probability[0] * 100:.2f}%")
        st.write(f"Default: {probability[1] * 100:.2f}%")


# ------------------------------------------------
# SAVE SECTION
# ------------------------------------------------

    if st.session_state.prediction is not None:

        st.subheader("Save Customer Data")

        file_name = st.text_input(
            "Enter CSV file name:",
            placeholder="customers.csv"
        )

        if st.button("Save"):

            if not file_name:

                st.warning("Please enter CSV file name.")

            else:

                if not file_name.endswith(".csv"):
                    file_name += ".csv"

                data = {
                    "Serial No.": [st.session_state.ID],
                    "Name": [name],
                    "Phone No.": [phone],
                    "Email": [email],
                    "default.payment.next.month": [st.session_state.prediction]
                }

                new_data = pd.DataFrame(data)

                if os.path.exists(file_name):

                    new_data.to_csv(
                        file_name,
                        mode="a",
                        header=False,
                        index=False
                    )

                    st.success(
                        f"Data successfully added to {file_name}"
                    )

                else:

                    new_data.to_csv(
                        file_name,
                        mode="w",
                        header=True,
                        index=False
                    )

                    st.success(
                        f"{file_name} created successfully!"
                    )   

def analysis():

    st.title("📊 Credit Card Default Analysis")



    csv_files = [
        file for file in os.listdir(".")
        if file.endswith(".csv")
    ]

    if not csv_files:

        st.warning("No CSV files found.")

        return


    selected_file = st.selectbox(
        "Select CSV File:",
        csv_files
    )

    # Read CSV
    default_data = pd.read_csv(selected_file)

    st.success(f"Loaded file: {selected_file}")


    target_column = "default.payment.next.month"

    if target_column not in default_data.columns:

        st.error(
            f"'{target_column}' column not found in CSV file."
        )

        return


    total_customers = len(default_data)

    default_customers = (
        default_data[target_column] == 1
    ).sum()

    non_default_customers = (
        default_data[target_column] == 0
    ).sum()

    default_rate = (
        default_customers / total_customers
    ) * 100 if total_customers > 0 else 0



    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            total_customers
        )

    with col2:
        st.metric(
            "Default Customers",
            default_customers
        )

    with col3:
        st.metric(
            "Non-Default Customers",
            non_default_customers
        )

    with col4:
        st.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

    st.divider()


    st.subheader("Default Distribution")

    chart_data = pd.DataFrame({
        "Status": [
            "Default",
            "Non-Default"
        ],
        "Customers": [
            default_customers,
            non_default_customers
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            chart_data.set_index("Status")
        )

    with col2:

        st.write("Customer Distribution")

        st.dataframe(
            chart_data,
            use_container_width=True
        )

    st.divider()


    st.subheader("Customer Data")

    st.dataframe(
        default_data,
        use_container_width=True
    )

pg = st.navigation([home, Prediction, analysis])
pg.run()

